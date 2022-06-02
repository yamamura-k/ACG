import torch

from utils import setup_logger

logger = setup_logger(__name__)
import numpy as np

from utils.cluster_coef_c.cluster_coef_2 import (
    compute_cluster_coef_from_distance_matrix_batch,
)


@torch.inference_mode()
class DI(object):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.params = config.di.params
        self.epsilon = config.eps

        self.visited_points = None  # size=[bs, iteration, dimension]
        self.sqdim = None  # sqrt(dimension)
        self.max_threshold = None  # diagonal length of search space

        self.D = None  # size=[bs, iteration, iteration]

        self.bs = None
        self.diversity_index = list()
        self.diversity_index_per_restart = list()
        self.mean_cluster_coefs = list()
        self.step_sizes = list()  #

        self.num_nodes = self.params.num_nodes

    def _update_distance(self, iteration, new_point, save_step=True):
        """()
        Parameters
        ---
            n_restart : int
                
            new_point : torch.tensor


        Returns
        ---
            None

        Note
        ---
        """
        self.bs = new_point.shape[0]
        _new_point = (
            new_point.detach()
            .view(-1, self.bs)
            .unsqueeze(0)
            .permute(2, 0, 1)
            .cpu()
            .clone()
        )
        if self.sqdim is None:
            self.sqdim = _new_point.shape[-1] ** 0.5
            self.max_cluster_coef = 2 * self.epsilon * self.sqdim
        if iteration == 0:
            self.update_cnt = torch.zeros(
                self.bs,
            ).type(torch.long)
            self.indices = torch.arange(self.bs).type(torch.long)
            self.visited_points = _new_point  # bs, iteration, dimension
            self.D = torch.zeros(size=(self.bs, 1, 1))

            if save_step:
                self.step_sizes.append(list())
        elif self.visited_points.shape[1] < self.num_nodes:
            _shape = self.visited_points.shape
            incoming_change = (
                (self.visited_points - _new_point.broadcast_to(size=_shape))
                .pow(2)
                .sum(dim=-1)
                .sqrt()
                .unsqueeze(-1)
            )
            self.D = torch.cat([self.D, incoming_change], dim=-1)
            incoming_change = torch.cat(
                [incoming_change, torch.zeros(size=(self.bs, 1, 1))], dim=1
            ).permute(0, 2, 1)
            self.D = torch.cat([self.D, incoming_change], dim=1)
            self.visited_points = torch.cat([self.visited_points, _new_point], dim=1)
        else:
            _shape = self.visited_points.shape
            self.visited_points[self.indices, self.update_cnt, :] = _new_point.squeeze()
            incoming_change = (
                (self.visited_points - _new_point.broadcast_to(size=_shape))
                .pow(2)
                .sum(dim=-1)
                .sqrt()
            )
            self.D[self.indices, self.update_cnt, :] = incoming_change
            self.D[self.indices, :, self.update_cnt] = incoming_change

        if hasattr(self, "before_points"):
            delta = (new_point.cpu() - self.before_points.cpu()).reshape(self.bs, -1)
            _l1norm = torch.norm(delta, p=1, dim=1)
            _l2norm = torch.norm(delta, p=2, dim=1)
            self.l1norm_deltax[iteration, :] = _l1norm.clone()
            self.l2norm_deltax[iteration, :] = _l2norm.clone()
        else:
            self.l1norm_deltax = torch.zeros(
                (self.config.solver.params.max_iter, self.bs)
            )
            self.l2norm_deltax = torch.zeros(
                (self.config.solver.params.max_iter, self.bs)
            )
        self.before_points = new_point.cpu().clone()

        self.update_cnt += 1
        self.update_cnt %= self.num_nodes

    def reset(self):
        del self.before_points
        del self.l1norm_deltax
        del self.l2norm_deltax

    def _get_cluster_coef(self):
        bs, n, _ = self.D.shape
        cluster_coef = np.zeros((bs,), dtype=np.float64)
        compute_cluster_coef_from_distance_matrix_batch(
            bs,
            n,
            self.max_cluster_coef,
            self.D.numpy().astype(np.float32).reshape(-1),
            cluster_coef,
        )
        average_coef = torch.from_numpy(cluster_coef)
        self.diversity_index.append(1 - average_coef)
        self.diversity_index_per_restart.append(1 - average_coef)
        return average_coef


    def getDI(self, xk, iteration, restart, step_size, *args, **kwargs):
        self._update_distance(iteration, xk)
        if isinstance(step_size, float):
            self.step_sizes[restart].append(
                torch.full(size=(xk.shape[0],), fill_value=step_size)
            )
        elif isinstance(step_size, torch.Tensor):
            assert len(step_size.shape) == 4
            step_size = step_size.squeeze(1).squeeze(1).squeeze(1)
            self.step_sizes[restart].append(step_size.cpu().clone())
        else:
            assert False
        if self.D.shape[1] == self.num_nodes:
            self._get_cluster_coef()
        return step_size, None

    def getClusterCoefPerRestart(self):

        if len(self.diversity_index) > 0:
            diversity_index_per_restart = torch.vstack(
                self.diversity_index_per_restart
            )
        else:
            max_iter = self.config.solver.params.max_iter
            diversity_index_per_restart = (
                torch.empty((max_iter - self.num_nodes, self.bs)) * torch.nan
            )

        self.diversity_index_per_restart = list()
        return diversity_index_per_restart