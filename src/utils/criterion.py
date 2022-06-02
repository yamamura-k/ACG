import torch.nn as nn
import numpy as np

from utils import setup_logger

logger = setup_logger(__name__)


class Loss:
    def getName(self):
        """
        Returns
        -------
        name: str
            loss name
        """
        return self.name

    def getTarget(self):
        """return attack-target label, if it is untarget, then return None.

        Returns
        -------
        None or int (label)
        """
        return None


class UntargetedLoss(Loss):
    pass


class TargetedLoss(Loss):
    pass


class CrossEntropy(UntargetedLoss):
    def __init__(self, params):
        self.name = "cross_entropy"
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False, reduction="mean")

    def __call__(self, logits, y_true, *args, **kwargs):
        """
        logits: tensor

        y_true : tensor

        """
        return self.cross_entropy(logits, y_true)


class Targeted_CrossEntropy(TargetedLoss):
    def __init__(self, params):
        self.name = "targeted_cross_entropy"
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False, reduction="mean")

    def __call__(self, logits, y_adv, *args, **kwargs):
        """
        y_adv: tensor

        """
        return self.cross_entropy(logits, y_adv)


class DLRLoss(UntargetedLoss):
    """
    See Also
    --------
    https://arxiv.org/pdf/2003.01690.pdf
    """

    def __init__(self, params):
        self.name = "dlrloss"

    def __call__(self, logits, y_true, *args, **kwargs):
        return dlr_loss(x=logits, y=y_true)


class Targeted_DLRLoss(TargetedLoss):
    """
    See Also
    --------
    https://arxiv.org/pdf/2003.01690.pdf
    """

    def __init__(self, params):
        self.name = "targeted_dlrloss"

    def __call__(self, logits, y, y_adv, *args, **kwargs):
        """ """
        return dlr_loss_targeted(x=logits, y=y, y_target=y_adv)


class CWLoss(TargetedLoss):
    """
    See Also
    --------

    """

    def __init__(self, params):
        self.name = "cwloss"

    def __call__(self, logits, y_true, output_target_label=False, *args, **kwargs):
        return cw_loss(x=logits, y=y_true, output_target_label=output_target_label)


def dlr_loss(x, y):
    """
    Parameters
    ----------
    x: torch.tensor
        output of model; shape is (n_batch, n_output)
    y: tensor


    Notes
    -----
    DLR = -\frac{z_y - \max_{i\neq y}z_i}{z_{\pi_1} - z_{\pi_3}}
    """
    x_sorted, ind_sorted = x.sort(dim=1)

    # xy1, 0.
    ind = (ind_sorted[:, -1] == y).float()

    z_y = x[np.arange(x.shape[0]), y]

    # ind=1 (True)→ argmax_{i\neq y}(x) = -2
    # ind=0 (False)→ argmax_{i\neq y}(x) = -1
    max_zi = x_sorted[:, -2] * ind + x_sorted[:, -1] * (1.0 - ind)

    # value_true_maximum = x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)
    value_true_maximum = z_y - max_zi

    pi1_pi3 = x_sorted[:, -1] - x_sorted[:, -3] + 1e-6

    loss = (-1.0) * value_true_maximum / pi1_pi3
    return loss.reshape(-1)


def dlr_loss_targeted(x, y, y_target):
    """
    Parameters
    ----------
    x: torch.tensor
        output of model; shape is (n_batch, n_output)
    y: tensor

    y_target: tensor


    Returns
    -------
    loss: tensor; shape is (torch.Size([n_batch])


    Notes
    -----
    Targeted_DLR = - \frac{z_y - z_t}{z_{\pi_1} - (z_{\pi_3})}
    """
    x_sorted, ind_sorted = x.sort(dim=1)

    #
    zy_zt = x[np.arange(x.shape[0]), y] - x[np.arange(x.shape[0]), y_target]

    denominator = x_sorted[:, -1] - 0.5 * x_sorted[:, -3] - 0.5 * x_sorted[:, -4] + 1e-6

    loss = zy_zt / denominator
    return loss.reshape(-1)


def cw_loss(x, y, output_target_label=False):
    """
    Parameters
    ----------
    x: torch.tensor
        output of model; shape is (n_batch, n_output)
    y: tensor

    output_target_label: bool
        if it is true, then return targeted label

    Returns
    -------
    loss: torch.tensor
        shape is (torch.Size([n_batch])

    Notes
    -----
      -\frac{z_y - \max_{i\neq y}z_i}
    """
    x_sorted, ind_sorted = x.sort(dim=1)

    # xy1, 0.
    ind = (ind_sorted[:, -1] == y).float()

    z_y = x[np.arange(x.shape[0]), y]

    # ind=1 (True)→ argmax_{i\neq y}(x) = -2
    # ind=0 (False)→ argmax_{i\neq y}(x) = -1
    max_zi = x_sorted[:, -2] * ind + x_sorted[:, -1] * (1.0 - ind)

    # value_true_maximum = x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)
    value_true_maximum = z_y - max_zi
    loss = (-1.0) * value_true_maximum

    if output_target_label:
        target = ind * ind_sorted[:, -2] + (1 - ind) * ind_sorted[:, -1]
        return loss.reshape(-1), target
    else:
        return loss.reshape(-1)


def get_criterion(name, params=None, *args, **kwargs):
    """
    Parameters
    ----------
    name: str

    params: attrdict or None
        

    Returns
    -------
    criterion: object

    """
    logger.info(f"\n [CRITERION]: {name}")
    if name == "ce":
        criterion = CrossEntropy(params)
    elif name == "targeted_ce":
        criterion = Targeted_CrossEntropy(params)
    elif name == "dlr":
        criterion = DLRLoss(params)
    elif name == "targeted_dlr":
        criterion = Targeted_DLRLoss(params)
    elif name == "cw":
        criterion = CWLoss(params)
    elif name == "dynamic":
        criterion = DynamicLoss(params)
    else:
        raise ValueError(f"unknowkn loss {name}")

    return criterion
