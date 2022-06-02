import json
import platform
import collections.abc as container_abcs
import pickle
from collections import defaultdict

import torch
import yaml
from attrdict import AttrDict

from utils import setup_logger

logger = setup_logger(__name__)


class Clamp:
    """
    Paramters
    ---------
    upper:
        ()

    lower:
        ()

    device:

    """

    def __init__(self, upper, lower, device=torch.device("cpu")):
        # TODO: L2 norm.
        self.device = device
        self.upper = upper
        self.lower = lower

        assert (self.upper > self.lower).all().item()

    def __call__(self, x_adv, ind=None):
        """
        Parameters
        ----------
        x_adv: tensor
            .
        idx: int or None
            .

        Returns
        -------
        x_adv:
        """
        if ind is None:
            assert x_adv.shape == self.upper.shape

            x_adv = torch.minimum(torch.maximum(x_adv, self.lower), self.upper)

            assert (x_adv >= self.lower).all().item()
            assert (x_adv <= self.upper).all().item()

        else:
            assert x_adv.shape == self.upper[ind].shape
            x_adv = torch.minimum(
                torch.maximum(x_adv, self.lower[ind]), self.upper[ind]
            )

            assert (x_adv >= self.lower[ind]).all().item()
            assert (x_adv <= self.upper[ind]).all().item()
        return x_adv


def read_paramfile(param_path):
    """set attributes from param file

    Parameters
    ----------
    param_path : str
        parameter file path
    """
    param_dict = dict()
    for line in open(param_path, "r"):
        line = line.strip()
        if not line:
            continue

        # comment out check
        if line[0] == "#":
            continue

        # read parameters
        param_name, param_type, param_value = line.split()
        if param_value in {"infty", "unlimited"}:
            param_value = float("inf")
        elif param_type == "bool":
            param_value = bool(eval(param_value))
        elif param_type == "int":
            param_value = int(param_value)
        elif param_type == "float":
            param_value = float(param_value)
        elif param_type != "str":
            message = (
                "Param File Error!!",
                f"the format `{line}` is not correct.",
                "param_type must be select from int, float and str.",
                "e.g. epsilon float 0.1",
            )
            logger.error("\n".join(message))
        param_dict[param_name] = param_value
        logger.info(f"set {param_name} <-- {param_value}")
    return param_dict


def read_yaml(paths):
    """reader of yaml file

    Parameters
    ----------
    paths: str or list of str
        yaml file path(s)
    """
    if isinstance(paths, str):
        paths = [paths]

    obj = dict()
    for path in paths:
        logger.debug(f"\n [ READ ] {path}")
        f = open(path, mode="r")
        _obj = yaml.safe_load(f)
        f.close()

        for key, value in _obj.items():
            logger.info(f"\n {key} ← {value}")
        obj.update(_obj)
    return obj


def overwrite_config(cmd_param, config):
    """update parameter dict

    Parameters
    ----------
    cmd_param : list of str
        each element is param:cast:value
    config : dict
        parameter dict
    """
    for param_set in cmd_param:
        param, cast, value = param_set.split(":")
        param_layer = param.split(".")
        param_dict = config
        update = True
        for param_name in param_layer[:-1]:
            if param_name not in param_dict:
                param_dict[param_name] = dict()
                update = False
            param_dict = param_dict[param_name]
        if param_layer[-1] not in param_dict:
            update = False
        if cast == "bool":
            assert value in {
                "True",
                "False",
            }, "the param type bool must be True or False."
            param_dict[param_layer[-1]] = False if value == "False" else True
        else:
            param_dict[param_layer[-1]] = eval(f'{cast}("{value}")')
        logger.info(f'{["new", "update"][update]} param {param_layer[-1]} <- {value}')
    return AttrDict(config)


def defaultdict2dict(ddict):
    """defaultdict, dict.
    Parameters
    ----------
    ddict: defaultdict or dict

    Returns
    -------
    dict_ddict: dict
    """
    if isinstance(ddict, defaultdict) or isinstance(ddict, dict):
        ddict = {k: defaultdict2dict(v) for k, v in ddict.items()}
    return ddict


def mk_defaultdict(dep):
    """ """
    if dep == 1:
        return dict()
    else:
        return defaultdict(
            lambda: mk_defaultdict(
                dep - 1,
            )
        )


class CalcRadius(object):
    """.

    Parameters
    ----------
    x_nat: tensor

    epsilon: float


    Notes
    -----

    Returns
    -------
    radius: tensor


    """

    def __init__(
        self, x_nat, epsilon, device=torch.device("cpu"), *args, **kargs
    ) -> None:
        super().__init__()

        self.epsilon = epsilon
        self.device = device

        # L\inftylower bound
        self.lower = torch.clamp(x_nat - epsilon, 0, 1).to(self.device)
        self.upper = torch.clamp(x_nat + epsilon, 0, 1).to(self.device)

    def __call__(self, xk, *args, **kwargs):
        """ """
        radius = torch.min(
            torch.min(torch.abs(xk - self.lower)),
            torch.min(torch.abs(self.upper - xk)),
        )
        assert radius.item() >= 0
        return radius

    def __str__(self):
        s = f"[RADIUS]\n"
        s += f"epsilon: {self.epsilon}"
        s += f"device : {self.device}"

        return s


def CheckInNeighborhood(x, clamp, ind=None):
    """
    Parameters
    ----------
    x: tensor


    clamp: Clamp
        function

    Returns
    -------
    is_in: bool
        True→, False:

    Notes
    -----
    https://pytorch.org/docs/stable/generated/torch.logical_not.html
    """
    assert x.shape == clamp.lower.shape
    if len(x.shape) == 4:
        sum_dim = (1, 2, 3)
    else:
        sum_dim = 1
    clamped_x = clamp(x)
    ind_in = (clamped_x != x).sum(dim=sum_dim) == 0.0
    ind_over = torch.logical_not(ind_in)

    ## FIXME:
    if ind_over.any().item():
        is_in = False
    else:
        is_in = True

    return is_in, ind_over, ind_in


def read_picklefile(pickle_path):
    logger.info(f"\n [READ]: {pickle_path}")
    with open(pickle_path, mode="rb") as f:
        data = pickle.load(f)

    return data


class SumBoundPoints(object):
    """ """

    def __init__(
        self, x_nat, epsilon, device=torch.device("cpu"), *args, **kargs
    ) -> None:
        super().__init__()

        self.epsilon = epsilon
        self.device = device

        # L\inftylower bound
        self.lower = torch.clamp(x_nat - epsilon, 0, 1).to(self.device)
        self.upper = torch.clamp(x_nat + epsilon, 0, 1).to(self.device)
        self.bs = x_nat.shape[0]

    def __call__(self, xk, *args, **kwargs):
        """ """
        num_lower = torch.where(
            xk.reshape(self.bs, -1) == self.lower.reshape(self.bs, -1)
        )[1].shape[0]
        num_upper = torch.where(
            xk.reshape(self.bs, -1) == self.upper.reshape(self.bs, -1)
        )[1].shape[0]

        return num_lower, num_upper

    def __str__(self):
        s = f"[SumBoundPoints]\n"
        s += f"epsilon: {self.epsilon}"
        s += f"device : {self.device}"

        return s


class CountBoundPoints(object):
    """ """

    def __init__(
        self, x_nat, epsilon, device=torch.device("cpu"), *args, **kargs
    ) -> None:
        super().__init__()

        self.epsilon = epsilon
        self.device = device

        # L\inftylower bound
        self.lower = torch.clamp(x_nat - epsilon, 0, 1).to(self.device)
        self.upper = torch.clamp(x_nat + epsilon, 0, 1).to(self.device)
        self.bs = x_nat.shape[0]

    def __call__(self, xk, *args, **kwargs):
        """ """
        num_lower = (
            (xk.reshape(self.bs, -1) == self.lower.reshape(self.bs, -1))
            .sum(dim=1)
            .cpu()
        )
        num_upper = (
            (xk.reshape(self.bs, -1) == self.upper.reshape(self.bs, -1))
            .sum(dim=1)
            .cpu()
        )

        return num_lower, num_upper

    def __str__(self):
        s = f"[CountBoundPoints]\n"
        s += f"epsilon: {self.epsilon}"
        s += f"device : {self.device}"

        return s


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)


class LocalOpt(object):
    localopts = defaultdict(list)

    def __init__(self, bs):
        super().__init__()
        self.bs = bs

    def __call__(self, localopt, indices):
        """local opt"""
        lt_indices = torch.where(indices)[0].tolist()
        for i in lt_indices:
            self.localopts[i].append(localopt[i].unsqueeze(0))

    def getLocalOpts(self, index):
        """local opt"""
        assert isinstance(index, int)
        return self.localopts[index]

    def save(self, path: str):
        """localopt"""
        logger.info(f"[SAVE]: {path}")
        torch.save(self.localopts, f=path)

    def __str__(self):
        pass



def get_machine_info():
    system = platform.system()
    plf = platform.platform(aliased=True)
    node = platform.node()
    processor = platform.processor()

    return dict(system=system, platform=plf, node=node, processor=processor)


def output_execute_info(config, save_path):
    cmd_argv = config.cmd
    envirion_variables = config.environ_variables
    # git_hash = config.git_hash
    # branch = config.branch
    machine = config.machine

    execute_info = dict(
        cmd_argv=cmd_argv,
        # git_hash=git_hash,
        # branch=branch,
        machine=machine,
        envirion_variables=envirion_variables,
    )

    f = open(save_path, mode="w")
    json.dump(execute_info, fp=f, indent=2)
    f.close()
