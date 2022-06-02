from collections import defaultdict
import os
import glob
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from utils import setup_logger, mk_defaultdict

logger = setup_logger(__name__)


class StaticEvaluater(object):
    """ """

    dict_sum = dict()

    def __init__(self):
        super().__init__()

    def eval(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, result_dir):
        """ """
        assert type(result_dir) is str

        folders = [
            f
            for f in os.listdir(result_dir)
            if os.path.isdir(os.path.join(result_dir, f))
        ]

        self.result_dir = result_dir

        for idx in folders:
            ## picklehard code
            ## FIXME
            pt_files = glob.glob(f"{os.path.join(result_dir, idx)}/*pickle")
            assert len(pt_files) == 1
            pt_file = pt_files[0]
            self.dict_sum[idx] = self._read_pickle(pt_file)

    @staticmethod
    def _read_pickle(pt_file):
        """ """
        with open(pt_file, mode="rb") as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def _read_pth(pt_file):
        """ """
        data = torch.load(pt_file)
        return data

    def draw(self, x, y, name, idx, item):
        """ """
        # preprocess
        pt_output_root = os.path.join(f"{self.result_dir}_eval", item)
        if not os.path.isdir(pt_output_root):
            os.makedirs(pt_output_root)
        pt_output_path = os.path.join(pt_output_root, f"{name}.png")

        plt.plot(x, y)
        plt.title(name)
        logger.info(f"\n [SAVE]: {pt_output_path}")
        plt.savefig(pt_output_path)
        plt.clf()
        plt.close()