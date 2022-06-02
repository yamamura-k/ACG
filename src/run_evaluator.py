import os
from argparse import ArgumentParser
import csv
import json
import subprocess
from datetime import datetime

from utils import read_yaml
from checker import Evaluator
from checker import model_dict


def argparser():
    parser = ArgumentParser()

    parser.add_argument("-r", "--result", type=str, nargs="*", required=True)

    return parser


class EvaluatorException(Exception):
    def __init__(self):
        pass


def evaluate(root):
    yaml_path = os.path.join(root, "run.yaml")
    json_path = os.path.join(root, "execute_information.json")

    # ，architecture, citation
    yaml_data = read_yaml(yaml_path)
    model_name = yaml_data["model_name"]
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
        evaluator = Evaluator(p=1)
        attack_success_rate = f"{evaluator.load(root).item() * 100 :.2f}"
    except:
        raise EvaluatorException

    attacker_name = root.split("/")[-1]

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
                    data["git_hash"],
                    data["branch"],
                    date.strftime("%a %b %d %H:%M:%S %Y"),
                ],
            )
        )
    )


def main(directories):

    for root in sorted(directories, reverse=True):
        if not os.path.isdir(root):
            continue
        try:
            evaluate(root)
        except EvaluatorException:
            pass
        except:
            assert False


if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    directories = args.result
    main(directories=directories)
