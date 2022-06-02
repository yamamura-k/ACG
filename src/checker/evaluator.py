import os
import glob
import pickle
from collections import defaultdict
from argparse import ArgumentParser
from multiprocessing import Pool
from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import setup_logger

logger = setup_logger(__name__)


class Evaluator(object):
    """Attacker.
    Robust Value = The rate of successful attacks

    Examples
    --------
    >>> from checker.evaluator import Evaluator

    >>> evaluator = Evaluator()
    >>> evaluator.load('./tmp/PGDAttacker')
    >>> print(evaluator)
    result_dirs: ['./tmp/PGDAttacker']
    #sucess: 69
    #sample: 90
    RobustValue: 0.76667

    >>> evaluator = Evaluator(p=1)
    >>> evaluator.load('./tmp/LPAttacker')
    >>> print(evaluator)
    result_dirs: ['./tmp/LPAttacker']
    #sucess: 9
    #sample: 9
    RobustValue: 1.00000
    """

    result_dir = None
    dict_static = defaultdict(dict)

    def __init__(self, p):
        self.success = 0
        self.total_sample = 0
        self.result_dirs = []
        self.num_process = p

    def getRobustScore(self):
        return 1 - (self.success / self.total_sample if self.total_sample > 0 else 2)

    def _para_eval(self, folders):
        """, log.

        Parametes
        ---------
        folders: list
            .

        Returns
        -------
        lt_success: list
            (,  or ).
        """
        lt_success = list()
        if self.num_process > 1:
            with Pool(processes=self.num_process) as pool:
                i = pool.map(self._eval, folders)
                # tqdm(i, total=len(folders))
                lt_success += i
        else:
            for folder in folders:
                i = self._eval(folder=folder)
                lt_success.append(i)

        return lt_success

    def _eval(self, folder):
        """
        Parametrs
        ---------
        folder: str
            

        Returns:
            (folder, 0 or 1): tuple
                1, 0.

        """
        assert self.result_dir is not None
        pth_path = f"{os.path.join(self.result_dir, folder)}/logs.pth"
        logger.debug(pth_path)
        logs = torch.load(pth_path)

        ## preprocess
        if "adv_target" in logs.keys():
            adv_target = logs["adv_target"]
        else:
            print('[WARNING]: the log file does not have the key  " adv_target ".')
            print("adv_target=True.")
            adv_target = True
        #
        label = logs["info"]["label"]
        bs = len(label)

        if adv_target:
            for target in range(10):
                if target == label:
                    continue
                max_score_class = np.argmax(np.array(logs[target]["pred"]))
                if max_score_class == target:

                    return (folder, 1)
                else:
                    return (folder, 0)
        else:
            return (folder, torch.argmax(logs["attacked"]["pred"], dim=1) != label)

    def load(self, result_dir):
        """Attack

        Parameters
        ----------
        result_dir : str
            ...
        """
        assert type(result_dir) is str
        mnist_folders = [
            f
            for f in os.listdir(result_dir)
            if os.path.isdir(os.path.join(result_dir, f))
        ]
        self.result_dir = result_dir
        attacker_name = os.path.basename(result_dir)

        # summary of attack
        lt_success = sorted(self._para_eval(mnist_folders), key=lambda x: int(x[0]))

        array_success = [is_success.sum() for (_, is_success) in lt_success]
        # self.dict_static[attacker_name] = lt_success

        self.result_dirs.append(result_dir)
        self.success = sum(array_success)
        self.total_sample = sum([is_success.shape[0] for (_, is_success) in lt_success])
        return self.success / self.total_sample

    def show_success_rate(self, path):
        """.

        Paramters
        ---------
        path: str
            path
        """
        for attacker_name, lt_success in self.dict_static.items():
            lt_success_rate = self.get_success_rate(lt_success)
            plt.plot(lt_success_rate, label=attacker_name)

        plt.legend()
        plt.title("Success Rate")
        plt.xlabel("the number of queries")
        plt.ylabel(" success rate")
        logger.debug(f"\n [SAVE]: {path}")
        plt.savefig(path)
        plt.clf()
        plt.close()

    def get_success_rate(self, lt_success):
        """

        Parameters
        ----------
        lt_success: list
            query1, 0.
            tuple, (index, 0 or 1).

        Returns
        -------
        lt_sccess_rate: list
            query.
            0.
        """
        lt_success_rate = list()

        #
        counter = 0
        lt_success = sorted(lt_success, key=lambda x: int(x[0]))
        tmp = 0
        for idx, is_success in lt_success:
            # index0.
            assert is_success in {0, 1}
            counter += is_success
            # success_rate = counter / (int(idx) + 1)
            success_rate = counter / len(lt_success)
            lt_success_rate.append(success_rate)
            assert int(idx) >= tmp
            tmp = int(idx)

        return lt_success_rate

    def __str__(self):
        s = f"result_dirs: {self.result_dirs}\n"
        s += f"#success(↑): {self.success}\n"
        s += f"#sample:  {self.total_sample}\n"
        s += f"RobustScore(↓): {self.getRobustScore():.5f}"
        return s


def argparser():
    parser = ArgumentParser()

    parser.add_argument(
        "-p",
        "--result_dir",
        help="result directory name (not path)",
        required=True,
        type=str,
    )
    return parser


if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    evaluator = Evaluator()
    evaluator.load(args.result_dir)
    print(evaluator)
