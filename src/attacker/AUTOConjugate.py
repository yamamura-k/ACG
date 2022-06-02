import random
from collections import defaultdict

import numpy as np
import torch
from utils import setup_logger
from utils.step_size import update_step_size
from utils.common import Clamp

from attacker.Attacker import RestartAttacker

logger = setup_logger(__name__)

np.random.seed(0)
random.seed(0)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

def getBeta(
    gradn,
    gradn_1,
    sn_1,
    method="HS",
    use_clamp=True,
    replace_nan=False,
    *args,
    **kwargs,
):
    """
    Parameters
    ----------
    gradn: tensor

    gradn_1: tensor

    sn_1: tensor

    Returns
    -------
    betan: tensor

    Notes
    -----
    + Fletcher–Reeves
    + Polak-Ribiere
    """

    batch_size = gradn.shape[0]
    gradn = -gradn.reshape(batch_size, -1)
    gradn_1 = -gradn_1.reshape(batch_size, -1)
    sn_1 = sn_1.reshape(batch_size, -1)
    y = gradn - gradn_1
    logger.debug(f"method: {method}")
    if method == "FR":
        betan = gradn.pow(2).sum(dim=1) / (gradn_1.pow(2).sum(dim=1))
    elif method == "PR":
        betan = (gradn * y).sum(dim=1) / (gradn_1.pow(2).sum(dim=1))
    elif method == "HS":
        betan = -(gradn * y).sum(dim=1) / ((sn_1 * y).sum(dim=1))
    elif method == "DY":
        betan = -gradn.pow(2).sum(dim=1) / ((sn_1 * y).sum(dim=1))
    elif method == "HZ":
        betan = (
            (
                y
                - 2
                * sn_1
                * (
                    y.pow(2).sum(dim=1) / ((sn_1 * y).sum(dim=1)).unsqueeze(-1)
                )
            )
            * gradn
        ).sum(dim=1) / ((sn_1 * y).sum(dim=1))
    elif method == "DL":
        betan = ((y - sn_1) * gradn).sum(dim=1) / (
            (sn_1 * y).sum(dim=1)
        )
    elif method == "LS":
        betan = -(gradn * y).sum(dim=1) / ((sn_1 * gradn_1).sum(dim=1))
    elif method == "RMIL":
        betan = (gradn * y).sum(dim=1) / (sn_1.pow(2).sum(dim=1))
    elif method == "RMIL+":
        betan = (gradn * (y - sn_1)).sum(dim=1) / (
            sn_1.pow(2).sum(dim=1)
        )
    else:
        raise NotImplementedError
    if use_clamp:
        betan = betan.clamp(min=0)
    logger.debug(f"\n [BETA]: {betan}")
    if replace_nan:
        betan[torch.isnan(betan)] = 0.0
    return betan

class AUTOConjugate(RestartAttacker):
    """Auto-Projected Gradient Descent Attacker

    Parameters
    ----------
    epsilon : positive float
        radius of nearest
    step : int
        step value
    max_iter : int
        number of iteration
    num_sample : int
        number of sampling (initial  points)
    time_limit :  float
        time limit for solver
    random : bool
        initial point is random or not.
        If not, initial point is natural examples.

    Notes
    -----
    Croce, Francesco, and Matthias Hein.
    "Reliable evaluation of adversarial robustness
    with an ensemble of diverse parameter-free attacks."
    International Conference on Machine Learning. PMLR, 2020.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        epsilon = self.config.eps
        assert epsilon >= 0, f"Value of epsilon must be positive, but got {epsilon}."
        self.name = "AUTOConjugate"
        self.epsilon = epsilon
        self.gather.setParam("epsilon", epsilon)

        params = self.config.solver.params
        self.num_restarts = params.num_restart
        self.max_iter = params.max_iter
        self.eot_iter = params.eot_iter
        self.thr_decr = params.rho

        self.beta = params.beta
        self.momentum_alpha = params.momentum_alpha
        self.use_clamp = params.use_clamp

        self.replace_nan = params.replace_nan
        self.use_machine_epsilon = params.use_machine_epsilon
        self.activate_flag = params.activate_flag

        self.initial_point = params.initial_point
        self.restart_point = params.restart_point
        self.norm = self.config.norm

        self.time_limit = (
            params.time_limit if params.time_limit is not None else float("inf")
        )

        # criterion
        criterion_name = params.criterion.name
        self.criterion = self.get_criterion(
            name=criterion_name,
            params=params,
        )

        assert (
            not self.config.adv_target
        ), f"{self.name} is not supported specifying adversarial target."

    @torch.no_grad()
    def _attack_single_run(
        self, x, y, x_nat, restart, is_attacked_image_index, *args, **kwargs
    ):
        ## preprocess
        dict_static = defaultdict(list)

        y = y.to(self.device)

        self.n_iter_2, self.n_iter_min, self.size_decr = (
            max(int(0.22 * self.max_iter), 1),
            max(int(0.06 * self.max_iter), 1),
            max(int(0.03 * self.max_iter), 1),
        )
        # n_iter_2 determines when to check conditions to halve step sizes.
        message = f"parameters: {self.max_iter} {self.n_iter_2} {self.n_iter_min} {self.size_decr}"
        
        logger.debug(message)

        # iteration.
        bs = x.shape[0]
        succ_iter = torch.zeros((bs,))
        logits = self.model(x_nat.to(self.device))
        acc = logits.detach().argmax(dim=1) == y
        # -2.
        succ_iter[~acc] = -2

        #
        self.getBound(x_nat, self.epsilon)
        clamp = Clamp(upper=self.upper, lower=self.lower, device=self.device)

        if restart > 0:
            logger.info(f"\n [Restart From]: {self.restart_point}")
            if self.restart_point == "best":
                xk = x.clone()
            else:
                xk = self.getInitialPoint(
                    self.restart_point,
                    x_nat=x_nat,
                    index=torch.ones((bs,)).type(torch.bool),
                    is_attacked_image_index=is_attacked_image_index,
                    num_restart=restart * torch.ones((bs,)),
                )
        else:
            if self.initial_point == "input":
                xk = x_nat.clone().to(self.device)
            else:
                xk = self.getInitialPoint(
                    self.initial_point,
                    x_nat=x_nat,
                    index=torch.ones((bs,)).type(torch.bool),
                    is_attacked_image_index=is_attacked_image_index,
                    num_restart=restart * torch.ones((bs,)),
                )

        x_nat = x_nat.to(self.device)
        xk = xk.to(self.device)

        assert (xk >= 0).all().item()
        assert (xk <= 1).all().item()

        x_best = xk.clone()
        x_best_adv = xk.clone()

        loss_steps = torch.zeros([self.max_iter, x.shape[0]])
        loss_best_steps = torch.zeros([self.max_iter + 1, x.shape[0]])
        loss_steps_gather = torch.zeros([self.max_iter + 1, x.shape[0]])
        beta_steps = torch.zeros((self.max_iter + 1, bs))
        l1norm_sn_steps = torch.zeros((self.max_iter, bs))
        l2norm_sn_steps = torch.zeros((self.max_iter, bs))

        l1norm_grad_steps = torch.zeros((self.max_iter, bs))
        l2norm_grad_steps = torch.zeros((self.max_iter, bs))

        l1norm_xk_before_proj_steps = torch.zeros((self.max_iter, bs))
        l2norm_xk_before_proj_steps = torch.zeros((self.max_iter, bs))

        l1norm_before_proj_xk_steps = torch.zeros((self.max_iter, bs))
        l2norm_before_proj_xk_steps = torch.zeros((self.max_iter, bs))

        finding_attack_target_label_steps = torch.zeros((self.max_iter, bs))

        number_of_upper_elements_steps = torch.zeros((self.max_iter, bs))
        number_of_lower_elements_steps = torch.zeros((self.max_iter, bs))

        acc_steps = torch.zeros_like(loss_best_steps)

        xk.requires_grad_()
        grad = torch.zeros_like(xk)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(xk)  # 1 forward pass (eot_iter = 1)
                loss_indiv = self.criterion(logits, y)
                loss = loss_indiv.sum()

            # 1 backward pass (eot_iter = 1)
            grad += torch.autograd.grad(loss, [xk])[0].detach()
        grad /= float(self.eot_iter)
        grad_best = grad.clone()
        grad_best_1 = grad.clone()
        s_best_1 = grad.clone()

        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0  #

        # x00.
        succ_iter[torch.logical_and(succ_iter == 0, ~acc.cpu())] = -1

        loss_best = loss_indiv.detach().clone()
        loss_steps_gather[0] = loss_indiv.cpu().clone()
        step_size = (
            self.epsilon
            * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach()
            * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
        )  ##

        alpha = 2.0
        step_size = (
            alpha
            * self.epsilon
            * torch.ones([x_nat.shape[0], *([1] * len(x_nat.shape[1:]))])
            .to(self.device)
            .detach()
        )

        k = self.n_iter_2 + 0

        u = torch.arange(x_nat.shape[0], device=self.device)

        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)

        ##
        from utils.common import CountBoundPoints

        sum_bound_pointer = CountBoundPoints(
            x_nat=x_nat, epsilon=self.epsilon, device=self.device
        )

        sn_1 = None
        x_prev = xk.clone()
        for iteration in range(self.max_iter):
            gradn = grad.clone()
            xk = xk.detach()
            grad2 = xk - x_prev
            x_prev = xk.clone()
            if iteration == 0:
                sn = gradn
                a = 1.0
            else:
                a = self.momentum_alpha
                beta_n = (
                    getBeta(
                        gradn,
                        gradn_1,
                        sn_1,
                        method=self.beta,
                        use_clamp=self.use_clamp,
                        replace_nan=self.replace_nan,
                    )
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
                sn = gradn + beta_n * sn_1
                beta_steps[iteration + 1] = beta_n.squeeze().cpu().clone()

            l1norm_sn_steps[iteration, :] = torch.norm(
                sn.cpu().clone().reshape(bs, -1), p=1, dim=1
            )
            l2norm_sn_steps[iteration, :] = torch.norm(
                sn.cpu().clone().reshape(bs, -1), p=2, dim=1
            )

            sn_1 = sn.clone()
            gradn_1 = grad.clone()
            before_proj_1 = (xk + step_size * torch.sign(sn)).clone()
            xk_tmp = clamp(xk + step_size * torch.sign(sn))
            before_proj_2 = (xk + a * (xk_tmp - xk) + (1 - a) * grad2).clone()
            xk = clamp(xk + a * (xk_tmp - xk) + (1 - a) * grad2).detach()
            if a == 1.0:
                before_proj = before_proj_1
                assert torch.isclose(xk, xk_tmp).all()
            else:
                before_proj = before_proj_2
            l1norm_xk_before_proj_steps[iteration, :] = (
                (x_prev - before_proj).view((bs, -1)).abs().sum(dim=1)
            )
            l2norm_xk_before_proj_steps[iteration, :] = (
                (x_prev - before_proj).view((bs, -1)).pow(2).sum(dim=1).sqrt()
            )
            l1norm_before_proj_xk_steps[iteration, :] = (
                (xk - before_proj).view((bs, -1)).abs().sum(dim=1)
            )
            l2norm_before_proj_xk_steps[iteration, :] = (
                (xk - before_proj).view((bs, -1)).pow(2).sum(dim=1).sqrt()
            )

            num_lower, num_upper = sum_bound_pointer(xk)
            number_of_lower_elements_steps[iteration, :] = num_lower.clone()
            number_of_upper_elements_steps[iteration, :] = num_upper.clone()

            # gradient step
            xk.requires_grad_()
            grad = torch.zeros_like(xk)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(xk)  # 1 forward pass (eot_iter = 1)
                    loss_indiv = self.criterion(logits, y)
                    loss = loss_indiv.sum()
                # 1 backward pass (eot_iter = 1)
                grad += torch.autograd.grad(loss, [xk])[0].detach()
            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)

            _, ind_sorted = logits.sort(dim=1)
            correct_ind = (ind_sorted[:, -1] == y).float()
            cw_index = ind_sorted[:, -2] * correct_ind + ind_sorted[:, -1] * (
                1.0 - correct_ind
            )
            cw_index = cw_index.long().cpu()
            finding_attack_target_label_steps[iteration, :] = cw_index.clone()

            # , iteration + 1.
            # , succ_iter.
            succ_iter[acc] = iteration + 1

            acc_steps[iteration + 1] = acc + 0

            x_best_adv[(pred == 0).nonzero().squeeze()] = (
                xk[(pred == 0).nonzero().squeeze()] + 0.0
            )

            message = "\n iteration: {} - Loss: {:.6f} - Best loss: {:.6f}".format(
                iteration, loss.sum(), loss_best.sum()
            )
            logger.info(message)

            l1norm_grad_steps[iteration, :] = torch.norm(
                grad.cpu().clone().reshape(bs, -1), p=1, dim=1
            )
            l2norm_grad_steps[iteration, :] = torch.norm(
                grad.cpu().clone().reshape(bs, -1), p=2, dim=1
            )

            # check step size
            y1 = loss_indiv.detach().clone()
            loss_steps[iteration] = y1.cpu() + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = xk[ind].clone()
            grad_best[ind] = grad[ind].clone()
            grad_best_1[ind] = gradn_1[ind].clone()
            s_best_1[ind] = sn_1[ind].clone()

            loss_best[ind] = y1[ind] + 0
            loss_best_steps[iteration + 1] = loss_best + 0
            loss_steps_gather[iteration + 1] = y1.cpu() + 0

            counter3 += 1
            args = dict(
                u=u,
                xk=xk,
                x_best=x_best,
                grad=grad,
                gradn_1=gradn_1,
                sn_1=sn_1,
                grad_best=grad_best,
                grad_best_1=grad_best_1,
                s_best_1=s_best_1,
                step_size=step_size,
                iteration=iteration,
                counter3=counter3,
                k=k,
                loss_steps=loss_steps,
                loss_best=loss_best,
                loss_best_last_check=loss_best_last_check,
                reduced_last_check=reduced_last_check,
                thr_decr=self.thr_decr,
                size_decr=self.size_decr,
                n_iter_min=self.n_iter_min,
                activate_flag=self.activate_flag,
            )
            tmp = update_step_size(**args)
            (
                u,
                xk,
                x_best,
                grad,
                gradn_1,
                sn_1,
                step_size,
                counter3,
                k,
                loss_steps,
                loss_best,
                loss_best_last_check,
                reduced_last_check,
            ) = tmp
            ##
            dict_static["best_loss"].append(loss_best.sum().detach().cpu().item())
            dict_static["step_size"].append(step_size.cpu())

            ##
            ## FIXME:
            num_lower, num_upper = sum_bound_pointer(x_best)
            dict_static["num_lower"].append(num_lower)
            dict_static["num_upper"].append(num_upper)
            step_size_args = dict(
                xk=xk,
                step_size=step_size,
                iteration=iteration,
                restart=restart,
                count=None,
            )
            _, _ = self.di.getDI(**step_size_args)

        ##
        logits = self.model(x_best).cpu()
        loss_indiv = self.criterion(logits=logits, y_true=y.cpu())
        loss = loss_indiv.sum()
        logger.info(f"\n final loss:{loss:.3f}")
        acc = logits.argmax(dim=1) == y.cpu()
        # , max iteration + 1.
        succ_iter[acc] = self.max_iter + 1

        dict_static["from_lower"] = (x_best - self.lower).cpu()
        dict_static["from_upper"] = (self.upper - x_best).cpu()

        clsuter_coef_per_restart = self.di.getClusterCoefPerRestart()

        dict_static["cluster_coef"] = clsuter_coef_per_restart
        dict_static["acc"] = acc

        ## loss_best_steps
        dict_static["loss_best_steps"] = loss_best_steps[
            1:, :
        ].t()  # shape is (bs, n_iter)
        ## loss_steps
        dict_static["loss_steps"] = loss_steps_gather[
            1:, :
        ].t()  # shape is (bs, n_iter)

        dict_static["beta"] = beta_steps[1:, :].t()

        ## norm_deltax
        dict_static["l1norm_deltax"] = self.di.l1norm_deltax.clone()
        dict_static["l2norm_deltax"] = self.di.l2norm_deltax.clone()

        # success iteration
        dict_static["success_iteration"] = succ_iter

        # grad norm
        dict_static["l1norm_sn"] = l1norm_sn_steps.clone()
        dict_static["l2norm_sn"] = l2norm_sn_steps.clone()
        dict_static["l1norm_grad"] = l1norm_grad_steps.clone()
        dict_static["l2norm_grad"] = l2norm_grad_steps.clone()

        dict_static["l1norm_xk_prev_before_proj"] = l1norm_xk_before_proj_steps.clone()
        dict_static["l2norm_xk_prev_before_proj"] = l2norm_xk_before_proj_steps.clone()

        dict_static["l1norm_xk_before_proj"] = l1norm_before_proj_xk_steps.clone()
        dict_static["l2norm_xk_before_proj"] = l2norm_before_proj_xk_steps.clone()

        dict_static["number_of_upper_elements"] = number_of_upper_elements_steps.clone()
        dict_static["number_of_lower_elements"] = number_of_lower_elements_steps.clone()

        # target label
        dict_static[
            "finding_attack_target_label_steps"
        ] = finding_attack_target_label_steps.clone()

        self.gather.setParam(f"static{restart}", dict(dict_static))

        #
        self.lt_localopts.append(x_best.cpu())
        self.di.reset()

        return x_best.cpu()

    @torch.no_grad()
    def _attack_single_run_fast(
        self, x, y, x_nat, restart, is_attacked_image_index, *args, **kwargs
    ):

        y = y.to(self.device)

        self.n_iter_2, self.n_iter_min, self.size_decr = (
            max(int(0.22 * self.max_iter), 1),
            max(int(0.06 * self.max_iter), 1),
            max(int(0.03 * self.max_iter), 1),
        )
        message = f"parameters: {self.max_iter} {self.n_iter_2} {self.n_iter_min} {self.size_decr}"
        logger.debug(message)

        # iteration.
        bs = x.shape[0]
        logits = self.model(x_nat.to(self.device))

        self.getBound(x_nat, self.epsilon)
        clamp = Clamp(upper=self.upper, lower=self.lower, device=self.device)

        if restart > 0:
            logger.info(f"\n [Restart From]: {self.restart_point}")
            if self.restart_point == "best":
                xk = x.clone()
            else:
                xk = self.getInitialPoint(
                    self.restart_point,
                    x_nat=x_nat,
                    index=torch.ones((bs,)).type(torch.bool),
                    is_attacked_image_index=is_attacked_image_index,
                    num_restart=restart * torch.ones((bs,)),
                )
        else:
            if self.initial_point == "input":
                xk = x_nat.clone().to(self.device)
            else:
                xk = self.getInitialPoint(
                    self.initial_point,
                    x_nat=x_nat,
                    index=torch.ones((bs,)).type(torch.bool),
                    is_attacked_image_index=is_attacked_image_index,
                    num_restart=restart * torch.ones((bs,)),
                )

        x_nat = x_nat.to(self.device)
        xk = xk.to(self.device)

        assert (xk >= 0).all().item()
        assert (xk <= 1).all().item()

        x_best = xk.clone()
        x_best_adv = xk.clone()

        loss_steps = torch.zeros([self.max_iter, x.shape[0]])

        xk.requires_grad_()
        grad = torch.zeros_like(xk)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(xk)  # 1 forward pass (eot_iter = 1)
                loss_indiv = self.criterion(logits, y)
                loss = loss_indiv.sum()

            # 1 backward pass (eot_iter = 1)
            grad += torch.autograd.grad(loss, [xk])[0].detach()
        grad /= float(self.eot_iter)
        grad_best = grad.clone()
        grad_best_1 = grad.clone()
        s_best_1 = grad.clone()



        loss_best = loss_indiv.detach().clone()

        step_size = (
            self.epsilon
            * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach()
            * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
        )  ##

        alpha = 2.0
        step_size = (
            alpha
            * self.epsilon
            * torch.ones([x_nat.shape[0], *([1] * len(x_nat.shape[1:]))])
            .to(self.device)
            .detach()
        )

        k = self.n_iter_2 + 0

        u = torch.arange(x_nat.shape[0], device=self.device)

        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)

        sn_1 = None
        x_prev = xk.clone()
        for iteration in range(self.max_iter):
            gradn = grad.clone()
            xk = xk.detach()
            grad2 = xk - x_prev
            x_prev = xk.clone()
            if iteration == 0:
                sn = gradn
                a = 1.0
            else:
                a = self.momentum_alpha
                beta_n = (
                    getBeta(
                        gradn,
                        gradn_1,
                        sn_1,
                        method=self.beta,
                        use_clamp=self.use_clamp,
                        replace_nan=self.replace_nan,
                    )
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
                sn = gradn + beta_n * sn_1

            sn_1 = sn.clone()
            gradn_1 = grad.clone()
            xk_tmp = clamp(xk + step_size * torch.sign(sn))
            xk = clamp(xk + a * (xk_tmp - xk) + (1 - a) * grad2).detach()
            if a == 1.0:
                assert torch.isclose(xk, xk_tmp).all()

            # gradient step
            xk.requires_grad_()
            grad = torch.zeros_like(xk)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(xk)  # 1 forward pass (eot_iter = 1)
                    loss_indiv = self.criterion(logits, y)
                    loss = loss_indiv.sum()
                # 1 backward pass (eot_iter = 1)
                grad += torch.autograd.grad(loss, [xk])[0].detach()
            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y

            x_best_adv[(pred == 0).nonzero().squeeze()] = (
                xk[(pred == 0).nonzero().squeeze()] + 0.0
            )

            message = "\n iteration: {} - Loss: {:.6f} - Best loss: {:.6f}".format(
                iteration, loss.sum(), loss_best.sum()
            )
            logger.info(message)

            # check step size
            y1 = loss_indiv.detach().clone()
            loss_steps[iteration] = y1.cpu() + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = xk[ind].clone()
            grad_best[ind] = grad[ind].clone()
            grad_best_1[ind] = gradn_1[ind].clone()
            s_best_1[ind] = sn_1[ind].clone()

            loss_best[ind] = y1[ind] + 0


            counter3 += 1
            args = dict(
                u=u,
                xk=xk,
                x_best=x_best,
                grad=grad,
                gradn_1=gradn_1,
                sn_1=sn_1,
                grad_best=grad_best,
                grad_best_1=grad_best_1,
                s_best_1=s_best_1,
                step_size=step_size,
                iteration=iteration,
                counter3=counter3,
                k=k,
                loss_steps=loss_steps,
                loss_best=loss_best,
                loss_best_last_check=loss_best_last_check,
                reduced_last_check=reduced_last_check,
                thr_decr=self.thr_decr,
                size_decr=self.size_decr,
                n_iter_min=self.n_iter_min,
                activate_flag=self.activate_flag,
            )
            tmp = update_step_size(**args)
            (
                u,
                xk,
                x_best,
                grad,
                gradn_1,
                sn_1,
                step_size,
                counter3,
                k,
                loss_steps,
                loss_best,
                loss_best_last_check,
                reduced_last_check,
            ) = tmp

        logits = self.model(x_best).cpu()
        loss_indiv = self.criterion(logits=logits, y_true=y.cpu())
        loss = loss_indiv.sum()
        logger.info(f"\n final loss:{loss:.3f}")

        return x_best.cpu()

    def __str__(self):
        s = f"AUTOConjugate\n"
        s += f"     epsilon: {self.epsilon}\n"
        return s
