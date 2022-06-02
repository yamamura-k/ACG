import os
from argparse import ArgumentParser
import glob
import json
from datetime import datetime
import numpy as np
import pandas as pd

from utils import read_yaml
from checker import model_dict

dataset2num_instances = dict(cifar10=10000, cifar100=10000, imagenet=5000)


def argparser():
    parser = ArgumentParser()

    parser.add_argument("-r", "--result", type=str, nargs="*")
    parser.add_argument("-ns", "--n_restarts", type=int, default=5)

    return parser


class EvaluatorException(Exception):
    def __init__(self):
        pass


def evaluate(root, n_restarts):
    yaml_path = os.path.join(root, "run.yaml")
    json_path = os.path.join(root, "execute_information.json")

    # ，architecture, citation
    yaml_data = read_yaml(yaml_path)
    model_name = yaml_data["model_name"]
    if not model_name in model_dict.keys():
        print(root)
    assert model_name in model_dict.keys()
    citation = "\cite{" + model_dict[model_name][0] + "}"
    architecture = model_dict[model_name][1]

    # git
    with open(json_path, mode="r") as f:
        data = json.load(f)

    # 
    time = os.path.getmtime(json_path)
    date = datetime.fromtimestamp(time)

    # 
    try:
        loss_files = list(sorted(glob.glob(f"{root}/loss_restart*")))
        loss = (
            np.concatenate(
                [
                    np.expand_dims(
                        pd.read_csv(loss_file, index_col=False).to_numpy()[:, 1:], 1
                    )
                    for loss_file in loss_files
                ],
                axis=1,
            )
            if len(loss_files) > 0
            else []
        )
        num_instances = loss.shape[0]

        loss = loss[:, :n_restarts, :]
        attack_success_rate = (
            f"{((loss[:, :, -1].max(axis=1) >= 0).sum() / loss.shape[0]) * 100:.2f}"
        )
    except:
        raise EvaluatorException
    postfix = f"restart-{n_restarts}" if n_restarts != 5 else ""
    attacker_name = root.split("/")[-1] + postfix

    dataset = yaml_data["dataset"]
    assert num_instances == dataset2num_instances[dataset]

    print(
        ",".join(
            map(
                str,
                [
                    model_name,
                    attacker_name,
                    architecture,
                    citation,
                    attack_success_rate,
                    # data["git_hash"],
                    # data["branch"],
                    date.strftime("%a %b %d %H:%M:%S %Y"),
                ],
            )
        )
    )


def main(directories, n_restarts):

    for root in sorted(directories, reverse=True):
        if not os.path.isdir(root):
            # print(root)
            # print("skip")
            continue
        try:
            evaluate(root, n_restarts)
        except EvaluatorException:
            # print(root)
            # print("skip")
            pass
        except:
            print(root)
            assert False


if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    directories = args.result
    n_restarts = args.n_restarts
    main(directories=directories, n_restarts=n_restarts)
