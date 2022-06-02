import os
from collections import defaultdict

import torch
import matplotlib.pyplot as plt
import pandas as pd

from utils.logging import setup_logger

logger = setup_logger(__name__)



def plt_histgram(x, _range=None):
    """
    Parameters
    ----------
    x: np.array
        

    _range: None or tuple


    Returns
    -------

    Notes
    -----
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
    """
    fig, ax = plt.subplots(1)
    ax.hist(x=x, bins=50, range=_range)
    return fig, ax


def show_num_succ(save_path, max_iter, initial_succ=False, failed=False):
    """iteration.
    Parameters
    ----------
    save_path: str
        success_iter.csv

    max_iter: int


    initial_succ: bool


    failed: bool


    Returns
    -------
    fig: plt.figure

    """
    df = pd.read_csv(save_path)
    success_iteration = df["success_iteration"].to_numpy()
    x_max = max_iter + 1 if failed else max_iter
    x_min = -1 if initial_succ else 0
    _range = (x_min, x_max)

    fig, ax = plt_histgram(x=success_iteration, range=_range)

    #
    title = "Success Images per iteration"
    xlabel = "iteration"
    ylabel = "the number of success images"
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def save_loss2csv(tensor_loss, index_start, index_end, save_path):
    """

    Notes
    -----
    .
    """
    is_create = True
    if os.path.isfile(save_path):
        logger.info(f"\n [FIND]: {save_path}")
        is_create = False
    else:
        logger.info(f"\n [CREATE]: {save_path}")

    logger.info(f"\n [SAVE]: {save_path}")

    #
    assert len(tensor_loss.shape) == 2
    bs, max_iter = tensor_loss.shape
    logger.debug(f"\n batch_size:{bs}, iteration:{max_iter}")

    #
    f = open(save_path, mode="a")
    if is_create:
        header = ("image_index/iteration",) + tuple(str(i) for i in range(max_iter))
        string = ",".join(header)
        print(string, file=f)

    for j in range(bs):
        loss_per_image = tensor_loss[j, :]
        lt_loss_per_image = loss_per_image.tolist()
        string = ",".join(map(str, ([str(index_start + j)] + lt_loss_per_image)))
        print(string, file=f)

    # .
    f.close()


def save_succ_iter2csv(tensor_succ_iter, index_start, index_end, save_path):
    """

    Notes
    -----
    """
    is_create = True
    if os.path.isfile(save_path):
        logger.info(f"\n [FIND]: {save_path}")
        is_create = False
    else:
        logger.info(f"\n [CREATE]: {save_path}")

    logger.info(f"\n [SAVE]: {save_path}")

    #
    assert len(tensor_succ_iter.shape) == 1
    bs = tensor_succ_iter.shape[0]
    logger.debug(f"\n batch_size:{bs}")

    #
    f = open(save_path, mode="a")
    if is_create:
        header = ("image_index", "success_iteration")
        string = ",".join(header)
        print(string, file=f)

    for j in range(bs):
        succ_iter = int(tensor_succ_iter[j].item())
        string = ",".join([str(index_start + j), str(succ_iter)])
        print(string, file=f)

    # .
    f.close()


def experimental_post_process(
    config, gather, num_restart, root, index_start, index_end, batch_idx
):
    """."""
    info = gather.info
    for n in range(num_restart):
        try:
            dict_static = info[f"static{n}"]
        except KeyError:
            break

        loss_best_steps = dict_static["loss_best_steps"]

        file_name = f"loss_restart{n}.csv"
        save_path = f"{root}/{file_name}"
        save_loss2csv(
            tensor_loss=loss_best_steps,
            save_path=save_path,
            index_start=index_start,
            index_end=index_end,
        )

        loss_current_steps = dict_static["loss_steps"]
        file_name = f"current_loss_restart{n}.csv"
        save_path = f"{root}/{file_name}"
        save_loss2csv(
            tensor_loss=loss_current_steps,
            save_path=save_path,
            index_start=index_start,
            index_end=index_end,
        )

        cluster_coef = dict_static["cluster_coef"].t()
        file_name = f"cluster_coef_restart{n}.csv"
        save_path = f"{root}/{file_name}"
        save_loss2csv(
            tensor_loss=cluster_coef,
            save_path=save_path,
            index_start=index_start,
            index_end=index_end,
        )

        success_iteration = dict_static["success_iteration"]
        file_name = f"success_iteration_restart{n}.csv"
        save_path = f"{root}/{file_name}"
        save_succ_iter2csv(
            tensor_succ_iter=success_iteration,
            index_start=index_start,
            index_end=index_end,
            save_path=save_path,
        )

        if "beta" in dict_static.keys():
            beta = dict_static["beta"]
            file_name = f"conjugate_beta_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )
        if "l1norm_deltax" in dict_static.keys():
            beta = dict_static["l1norm_deltax"].t()
            file_name = f"l1norm_deltax_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )
        if "l2norm_deltax" in dict_static.keys():
            beta = dict_static["l2norm_deltax"].t()
            file_name = f"l2norm_deltax_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )

        if "l1norm_sn" in dict_static.keys():
            beta = dict_static["l1norm_sn"].t()
            file_name = f"l1norm_sn_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )
        if "l2norm_sn" in dict_static.keys():
            beta = dict_static["l2norm_sn"].t()
            file_name = f"l2norm_sn_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )

        if "l1norm_grad" in dict_static.keys():
            beta = dict_static["l1norm_grad"].t()
            file_name = f"l1norm_grad_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )

        if "l2norm_grad" in dict_static.keys():
            beta = dict_static["l2norm_grad"].t()
            file_name = f"l2norm_grad_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )

        if "l1norm_xk_prev_before_proj" in dict_static.keys():
            beta = dict_static["l1norm_xk_prev_before_proj"].t()
            file_name = f"l1norm_xk_prev_before_proj_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )

        if "l2norm_xk_prev_before_proj" in dict_static.keys():
            beta = dict_static["l2norm_xk_prev_before_proj"].t()
            file_name = f"l2norm_xk_prev_before_proj_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )

        if "l1norm_xk_before_proj" in dict_static.keys():
            beta = dict_static["l1norm_xk_before_proj"].t()
            file_name = f"l1norm_xk_before_proj_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )

        if "l2norm_xk_before_proj" in dict_static.keys():
            beta = dict_static["l2norm_xk_before_proj"].t()
            file_name = f"l2norm_xk_before_proj_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )

        if "l1norm_x_adv_old2before_proj_x_adv_1" in dict_static.keys():
            beta = dict_static["l1norm_x_adv_old2before_proj_x_adv_1"].t()
            file_name = f"1norm_x_adv_old2before_proj_x_adv_1_12_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )
        if "l2norm_x_adv_old2before_proj_x_adv_1" in dict_static.keys():
            beta = dict_static["l2norm_x_adv_old2before_proj_x_adv_1"].t()
            file_name = f"l2norm_x_adv_old2before_proj_x_adv_1_12_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )
        if "l1norm_x_adv_old2proj_x_adv_1" in dict_static.keys():
            beta = dict_static["l1norm_x_adv_old2proj_x_adv_1"].t()
            file_name = f"l1norm_x_adv_old2proj_x_adv_1_13_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )
        if "l2norm_x_adv_old2proj_x_adv_1" in dict_static.keys():
            beta = dict_static["l2norm_x_adv_old2proj_x_adv_1"].t()
            file_name = f"l2norm_x_adv_old2proj_x_adv_1_13_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )
        if "l1norm_before_proj_x_adv_12proj_x_adv_1" in dict_static.keys():
            beta = dict_static["l1norm_before_proj_x_adv_12proj_x_adv_1"].t()
            file_name = f"l1norm_before_proj_x_adv_12proj_x_adv_1_23_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )
        if "l2norm_before_proj_x_adv_12proj_x_adv_1" in dict_static.keys():
            beta = dict_static["l2norm_before_proj_x_adv_12proj_x_adv_1"].t()
            file_name = f"l2norm_before_proj_x_adv_12proj_x_adv_1_23_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )
        if "l1norm_before_proj_z2proj_z" in dict_static.keys():
            beta = dict_static["l1norm_before_proj_z2proj_z"].t()
            file_name = f"l1norm_before_proj_z2proj_z_24_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )
        if "l2norm_before_proj_z2proj_z" in dict_static.keys():
            beta = dict_static["l2norm_before_proj_z2proj_z"].t()
            file_name = f"l2norm_before_proj_z2proj_z_24_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )
        if "l1norm_proj_x_adv_12x_adv" in dict_static.keys():
            beta = dict_static["l1norm_proj_x_adv_12x_adv"].t()
            file_name = f"l1norm_proj_x_adv_12x_adv_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )
        if "l2norm_proj_x_adv_12x_adv" in dict_static.keys():
            beta = dict_static["l2norm_proj_x_adv_12x_adv"].t()
            file_name = f"l2norm_proj_x_adv_12x_adv_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=beta,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )

        if "finding_attack_target_label_steps" in dict_static.keys():
            finding_attack_target_label = dict_static[
                "finding_attack_target_label_steps"
            ].t()
            file_name = f"finding_attack_target_label_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=finding_attack_target_label,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )

        if "number_of_lower_elements" in dict_static.keys():
            finding_attack_target_label = dict_static["number_of_lower_elements"].t()
            file_name = f"number_of_lower_elements_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=finding_attack_target_label,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )

        if "number_of_upper_elements" in dict_static.keys():
            finding_attack_target_label = dict_static["number_of_upper_elements"].t()
            file_name = f"number_of_upper_elements_restart{n}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=finding_attack_target_label,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )

    if "step_sizes" in info.keys():
        step_sizes = info["step_sizes"]
        for restart in range(len(step_sizes)):
            step_size = torch.vstack(step_sizes[restart]).t()
            file_name = f"step_size_restart{restart}.csv"
            save_path = f"{root}/{file_name}"
            save_loss2csv(
                tensor_loss=step_size,
                save_path=save_path,
                index_start=index_start,
                index_end=index_end,
            )

    # ============
    # output log
    # ============
    for i, index in enumerate(range(index_start, index_end)):
        log_path = os.path.join(root, str(batch_idx), f"log_index{index}.txt")

        with open(log_path, mode="w") as f:

            msg = (
                info["start_msg"],
                f"param is " + ", ".join(config.yaml_paths),
                f"data  is " + config.dataset,
            )
            print("\n".join(msg), file=f)
            try:
                dict_static = info["static0"]
            except KeyError:
                break

            for n in range(num_restart):

                dict_static = info[f"static{n}"]
                loss_best_steps = dict_static["loss_best_steps"]
                loss_steps = dict_static["loss_steps"]
                num_nodes = config.di.params.num_nodes
                cluster_coef = dict_static["cluster_coef"].t()

                print(f"\n [ Restart{n} ]", file=f)
                msg = f"{' ':2} {'Loss':8}  {'BestLoss':9} {'ClusCoef':7}"
                print(msg, file=f)
                for iteration in range(loss_best_steps.shape[1]):
                    loss_best = loss_best_steps[i, iteration].item()
                    loss_step = loss_steps[i, iteration].item()
                    msg = f"{iteration + 1:2} {loss_step:.7} {loss_best:.7}"
                    if iteration >= num_nodes:
                        msg += f" {cluster_coef[i, iteration-num_nodes].item():.7}"
                    print(msg, file=f)

            print("\n", file=f)
            status = "Failed" if info["acc"][i].item() else "Success"
            msg = (
                f"Objective Value = {info['final_loss'][i].item()}",
                f"Status          = {status}",
                f"** Paramters **",
                f"epsilon       = {config.eps}",
                f"Iteration     = {config.solver.params.max_iter}",
                f"Restart       = {config.solver.params.num_restart}",
                f"Initial Point = {config.solver.params.initial_point}",
                f"Restart Point = {config.solver.params.restart_point}",
                f"Criterion     = {config.solver.params.criterion.name}",
                f"** Time **",
                f"Attack Time   = {info['time']}",
            )
            print("\n".join(msg), file=f)
            print(info["end_msg"], file=f)


class BaseViewer(object):
    dict_items = defaultdict(list)

    def __init__(self, *args, **kwargs):
        super().__init__()

    def output(self):
        return self.dict_items

    def output_pervalues(self, key):
        assert key in self.dict_items.keys()
        return self.dict_items[key]

    def output_diagram(self, save_path, start_index=None):
        for k, v in self.dict_items.items():
            if start_index is not None:
                assert isinstance(start_index, int)
                v = v[start_index:]

            plt.plot(v, label=k)

        plt.legend()
        logger.info(f"[SAVE]: {save_path}")
        plt.savefig(save_path)
        plt.clf()
        plt.close()


class Exp1Viewer(BaseViewer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def add(
        self, target_l2norm, target_linfnorm, move_l2norm, move_linfnorm, objective
    ):

        keys = [
            "target_l2norm",
            "target_linfnorm",
            "move_l2norm",
            "move_linfnorm",
            "objective",
        ]
        values = [target_l2norm, target_linfnorm, move_l2norm, move_linfnorm, objective]
        for k, v in zip(keys, values):
            self.dict_items[k].append(v)
