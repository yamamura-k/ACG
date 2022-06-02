import numpy as np


def update_step_size(
    u,
    xk,
    x_best,
    grad,
    gradn_1,
    sn_1,
    grad_best,
    grad_best_1,
    s_best_1,
    step_size,
    iteration,
    counter3,
    k,
    loss_steps,
    loss_best,
    loss_best_last_check,
    reduced_last_check,
    thr_decr,
    size_decr,
    n_iter_min,
    activate_flag=False,
):
    if counter3 == k:
        # case: norm in {"Linf", "L2"}:
        fl_oscillation = check_oscillation(
            loss_steps.detach().cpu().numpy(),
            iteration,
            k,
            loss_best.detach().cpu().numpy(),
            k3=thr_decr,
        )

        did_not_move = (grad == gradn_1).all(dim=1).all(dim=1).all(
            dim=1
        ).cpu().numpy() * activate_flag
        fl_oscillation = ((fl_oscillation + did_not_move) / 2).astype(bool)

        fl_reduce_no_impr = (~reduced_last_check) * (
            loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy()
        )
        fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
        reduced_last_check = np.copy(fl_oscillation)
        loss_best_last_check = loss_best.clone()

        if np.sum(fl_oscillation) > 0:
            step_size[u[fl_oscillation]] /= 2.0
            n_reduced = fl_oscillation.astype(float).sum()

            fl_oscillation = np.where(fl_oscillation)

            xk[fl_oscillation] = x_best[fl_oscillation].clone()
            grad[fl_oscillation] = grad_best[fl_oscillation].clone()
            gradn_1[fl_oscillation] = grad_best_1[fl_oscillation].clone()
            sn_1[fl_oscillation] = s_best_1[fl_oscillation].clone()

        k = np.maximum(k - size_decr, n_iter_min)

        counter3 = 0
    return (
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
    )


def check_oscillation(x, j, k, y5, k3=0.75):
    """
    Parameters
    ----------
    x: ndarray
        loss of criterion, which shape is (the number of iteration, the number of batch).
    j: int
        iteration
    k: int
        checkpointiteration.
    y5: ndarray
        the best loss of criterion, which shape is (the number of batch)
    k3: float
        rho

    Returns
    -------

    """
    # condition1
    t = np.zeros(x.shape[1])
    for counter5 in range(k):
        # 1_{f(x^(i+1)) > f(x^(i))}
        t += x[j - counter5] > x[j - counter5 - 1]
    return t <= k * k3 * np.ones(t.shape)
