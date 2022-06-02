import os
import datetime
from time import time


import torch
import torch.nn.functional as F
from utils import read_yaml, setup_logger
from utils.criterion import get_criterion
from utils.compute_diversity_index import DI

logger = setup_logger(__name__)

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


class Attacker(object):
    """Constructor

    Attributes
    ----------
    model : callable objecct
        whose input is Tensor (1, X) shape and output is Tensor (1, 1) shape
    W : Tensor
        weight paramter
    bias : Tensor
        bias paramter
    bound_opt : BoundOptimizer
        optimizer of each bound of unit
    config: AttrDict
        attacker parameters
    """

    def __init__(
        self,
        model=None,
        experiment=False,
        config=None,
        gather=None,
        save_image=False,
        save_points=False,
        save_localopts=False,
        export_statistics=False,
        *args,
        **kwargs,
    ):

        self.all_start_time = time()

        self.model = model
        self.config = config
        self.gather = gather
        self.save_image = save_image
        self.experiment = experiment
        self.save_points = save_points
        self.save_localopts = save_localopts
        self.export_statistics = export_statistics

        # set device
        if config.gpu_id is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{config.gpu_id}")

        start_time = datetime.datetime.now()
        msg = f"Adversarial Attack start at [{start_time.strftime('%a %b %d %H:%M:%S %Y')}]"
        print(msg)
        self.gather.setParam(key="start_msg", value=msg)

    @torch.no_grad()
    def attack(self, *args, **kwargs):
        log = dict()

        elaps = -time()
        adv_example = self._attack(log=log, *args, **kwargs)
        elaps += time()

        log["time"] = elaps
        self.gather.setParam(key="time", value=elaps)

        if self.save_image:
            if adv_example is None:
                logger.warning("\n adv_example is None.")
                log["pred"] = None
                log["image"] = None
                return adv_example, log

            else:
                logger.info("n [SAVE] vulnerable image")
                log["image"] = adv_example.cpu()

        if self.model is not None:
            log["pred"] = F.softmax(
                self.model(adv_example.to(self.device)), dim=1
            ).cpu()

        else:
            log["pred"] = None

        logger.debug(log["pred"])

        end_time = datetime.datetime.now()
        msg = f"Adversarial Attack end at [{end_time.strftime('%a %b %d %H:%M:%S %Y')}]"
        print(msg)
        self.gather.setParam(key="end_msg", value=msg)

        return adv_example, log

    def readParamFile(self, param_path):
        """read and set attributets from parameter file

        Parameters
        ----------
        param_path : str
            parameter file path
        """

        param_dict = read_yaml(param_path)

        for param_name, param_value in param_dict.items():
            setattr(self, param_name, param_value)

    def get_criterion(self, name, *args, **kwargs):
        return get_criterion(name, *args, **kwargs)

    def getSuccessIndex(self, output, y):
        """index

        Parameters
        ----------
        output: torch.Tensor

        y: torch.Tensor


        Returns
        -------
        success_idx: torch.Tensor
            index
        """
        cw_loss = self.get_criterion(name="cw", params=self.config.solver.params)(
            output, y.to(self.device)
        )
        success_idx = torch.where(cw_loss > 0)[0]
        return success_idx

    def __repr__(self) -> str:
        return ""


class WhiteBoxAttacker(Attacker):
    """"""

    upper = None
    lower = None
    lt_localopts = list()

    def __init__(self, *args, **kwargs):
        """

        Notes
        -----
        https://pytorch.org/docs/stable/notes/randomness.html
        """
        super().__init__(*args, **kwargs)

    def _attack(self, x, y, *args, **kwargs):
        raise NotImplementedError

    def getInitialPoint(self, initial_point, index, x_nat=None, *args, **kwargs):
        """

        Parameters
        ----------
        initial_point: str

        """
        logger.debug(f"\n [initial point]: {initial_point}")
        if initial_point == "center":
            x_init = self.getCenter()
        elif initial_point == "random":
            x_init = self.getRandomInput(index=index, *args, **kwargs)
        elif initial_point == "random_autopgd":
            x_init = self.getRandomInput4AutoPGD(x_nat, index=index, *args, **kwargs)
        elif initial_point == "original":
            start_point_path = self.config.start_point_path
            data = torch.load(start_point_path)
            x_init = data["image"]
            iteration = data["iteration"]
            logger.info(f"\n [iteraiton]: {iteration}")
        elif initial_point == "ODI":
            x_init = self.getODSInitialize(x_nat=x_nat, index=index, *args, **kwargs)

        else:
            logger.warning(f"\n {initial_point} is not supported.")
            raise NotImplementedError

        assert (x_init >= self.lower[index].cpu()).all().item()
        assert (x_init <= self.upper[index].cpu()).all().item()
        return x_init

    @torch.no_grad()
    def getODSInitialize(
        self, x_nat, num_restart, is_attacked_image_index, *args, **kwargs
    ):
        """ODI"""
        assert hasattr(self.config.solver, "odi_params")

        def getVODS(x, wd):
            with torch.enable_grad():
                x.requires_grad_(True)
                logits = self.model(x)
                vector = (wd * logits).sum(dim=1)  # inner product
                grad = torch.autograd.grad(vector.sum(), x)[0].detach()

            # L2 normalization
            vods = grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
            return vods

        bs = x_nat.shape[0]

        ETA = self.config.solver.odi_params.step_size
        ITERATION = self.config.solver.odi_params.iteration

        assert ETA <= self.epsilon

        if self.config.dataset == "imagenet":
            output_dim = 1000
        elif self.config.dataset in {"cifar10", "mnist"}:
            output_dim = 10
        else:
            assert False
        w = torch.zeros((bs, output_dim)).to(self.device)

        target_image_index = self.getTargetImageIndex(
            is_attacked_image_index=is_attacked_image_index, *args, **kwargs
        )
        for i, image_index in enumerate(target_image_index):
            restart = num_restart[i].item()
            seed = image_index + restart
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            w[i] = (
                torch.FloatTensor(output_dim).uniform_(-1, 1).to(self.device)
            )  # cifar10hard code

        #
        x = x_nat.clone().to(self.device)
        for i in range(ITERATION):
            vods = getVODS(x, w)
            x = (x + ETA * torch.sign(vods)).clamp(min=self.lower, max=self.upper)
        return x.cpu()

    def getRandomInput(self, num_restart, is_attacked_image_index, *args, **kwargs):
        """"""
        width = self.upper - self.lower
        target_image_index = self.getTargetImageIndex(
            is_attacked_image_index=is_attacked_image_index, *args, **kwargs
        )
        perturb = torch.zeros_like(self.upper).to(self.device)
        for i, image_index in enumerate(target_image_index):

            restart = num_restart[i].item()
            # seedimage index + restart.
            seed = image_index + restart
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            perturb[i] = torch.rand(size=self.upper[i].shape).to(self.device)

        x_init = width * perturb + self.lower
        return x_init.cpu()

    def getCenter(self, *args, **kwargs):
        """"""
        center = (self.lower + self.upper) / 2
        return center.cpu()

    def getBound(self, x_nat, *args, **kwargs):
        """."""
        lower = torch.clamp(x_nat - self.epsilon, 0, 1).to(self.device)
        upper = torch.clamp(x_nat + self.epsilon, 0, 1).to(self.device)
        self.lower = lower
        self.upper = upper

    def getRandomInput4AutoPGD(self, x_nat, num_restart, index, *args, **kwargs):
        """
        Parameters
        ----------
        x_nat: tensor
        num_restart: int
        index: bool tensor
            True
        """
        # x_adv
        shape = self.lower[index].shape
        assert len(shape) in {2, 4}

        # =======================
        #
        target_index = self.getTargetImageIndex(index=index, *args, **kwargs)

        t = torch.zeros_like(x_nat)
        for i, image_index in enumerate(target_index):
            restart = num_restart[i].item()
            # seedimage index + restart.
            seed = image_index + restart
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            t[i] = 2 * torch.rand(x_nat[i].shape).detach() - 1
        # =======================
        x_adv = x_nat.detach() + self.epsilon * torch.ones(
            [x_nat.shape[0], 1, 1, 1]
        ).detach() * t / (
            t.reshape([t.shape[0], -1])
            .abs()
            .max(dim=1, keepdim=True)[0]
            .reshape([-1, 1, 1, 1])
        )
        x_adv = x_adv.clamp(0.0, 1.0)

        return x_adv.reshape(shape)

    def getTargetImageIndex(self, index, is_attacked_image_index, *args, **kwargs):
        """
        Parameters
        ----------
        is_attacked_image: tensor
        targetd_index: list
        """

        target_index = self.config.attacked_images[is_attacked_image_index][index]
        return target_index

    def __str__(self, *args, **kwargs):
        s = super().__str__()
        return s


class RestartAttacker(WhiteBoxAttacker):
    """

    Attributes
    ----------
    _attack:
    _attack_single_run:

    """

    num_restarts = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.di = DI(*args, **kwargs)

    @torch.no_grad()
    def _attack(self, x, y, *args, **kwargs):
        """

        Parameters
        ----------
        x: tensor

        y: tensor

        Returns
        -------
        adv_examples:

        """
        #
        adv_example = x.clone()
        y_clone = y.clone()
        self.lt_localopts = list()

        #
        if type(y) is int:
            y = torch.LongTensor([y])
        bs = x.shape[0]
        # acc
        logits = self.model(adv_example.to(self.device)).cpu()
        if self.experiment:
            acc = torch.ones((bs,)).type(torch.bool)
        else:
            acc = logits.max(dim=1)[1] == y_clone  # True

        logger.info(f"\n [Evaluator Function]: CW")
        evaluate_function = self.get_criterion(name="cw")

        best_evaluate_value = evaluate_function(logits, y_clone)

        for start in range(self.num_restarts):
            if acc.sum() == 0:
                continue

            xk = adv_example.clone()
            xk = xk[acc]
            yk = y_clone[acc]

            bs = xk.shape[0]
            logger.info(f"\n [RESTART{start}]: {bs}/{x.shape[0]}")

            if self.export_statistics:
                single_adv_example = self._attack_single_run(
                    x=xk,
                    y=yk,
                    x_nat=x[acc].clone(),
                    restart=start,
                    is_attacked_image_index=acc,
                    *args,
                    **kwargs,
                )

            else:
                single_adv_example = self._attack_single_run_fast(
                    x=xk,
                    y=yk,
                    x_nat=x[acc].clone(),
                    restart=start,
                    is_attacked_image_index=acc,
                    *args,
                    **kwargs,
                )

            # acc
            logits = self.model(single_adv_example.to(self.device)).cpu()
            acc_attacked = logits.max(dim=1)[1] == y_clone[acc]

            # restart
            evaluate_value = evaluate_function(logits, y_clone[acc])

            # acc = True
            is_best = evaluate_value > best_evaluate_value[acc]
            tmp_best_evaluate_value = best_evaluate_value[acc]
            tmp_best_evaluate_value[is_best] = evaluate_value[is_best].clone()
            best_evaluate_value[acc] = tmp_best_evaluate_value.clone()
            assert (
                (best_evaluate_value[acc][is_best] == evaluate_value[is_best])
                .all()
                .item()
            )

            tmp_adv_example = adv_example[acc]
            tmp_adv_example[is_best] = single_adv_example[is_best].clone()
            adv_example[acc] = tmp_adv_example.clone()
            assert (
                (adv_example[acc][is_best] == single_adv_example[is_best]).all().item()
            )

            if self.experiment:
                # accTrue.
                acc = torch.ones((bs,)).type(torch.bool)
            else:
                # False
                acc_clone = acc.clone()
                tmp_acc = acc_clone[acc_clone].clone()
                tmp_acc[~acc_attacked] = False
                acc[acc_clone] = tmp_acc.clone()

        if self.export_statistics:
            torch.save(
                self.di.diversity_index,
                os.path.join(self.config.result_dir, f"diversity_index_{self.name}.pth"),
            )
            torch.save(
                self.di.step_sizes,
                os.path.join(self.config.result_dir, f"step_size_{self.name}.pth"),
            )
            self.gather.setParam(f"step_sizes", self.di.step_sizes)
        if self.save_points:
            torch.save(
                self.di.visited_points,
                os.path.join(
                    self.config.result_dir, f"all_visited_point_{self.name}.pth"
                ),
            )
        if self.save_localopts:
            torch.save(
                self.lt_localopts,
                os.path.join(self.config.result_dir, f"localopts_{self.name}.pth"),
            )

        logits = self.model(adv_example.to(self.device)).cpu()
        loss = self.criterion(logits, y)
        acc = logits.max(dim=1)[1] == y_clone
        logger.info(f"\n [FINAL ACC]: {acc.sum()}/{x.shape[0]}")
        print(f"\n [ASR]: {x.shape[0] - acc.sum()}/{x.shape[0]}")
        adv_example = adv_example.cpu()
        assert (adv_example >= torch.clamp(x - self.epsilon, min=0)).all().item()
        assert (adv_example <= torch.clamp(x + self.epsilon, max=1)).all().item()

        self.gather.setParam(key="acc", value=acc)
        self.gather.setParam(key="final_loss", value=loss)
        return adv_example

    def _attack_single_run(self, *args, **kwargs):
        raise NotImplementedError

    def _attack_single_run_fast(self, *args, **kwargs):
        return self._attack_single_run(*args, **kwargs)

    def __str__(self, *args, **kwargs):
        s = super().__str__()
        return s
