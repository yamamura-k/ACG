# coding: UTF-8

import importlib
import math
import os
import shutil
import sys
import subprocess
from argparse import ArgumentParser

import git
import numpy as np
import torch
import yaml
from attrdict import AttrDict
from robustbench.data import load_imagenet
from robustbench.model_zoo import model_dicts as all_models
from robustbench.utils import load_model as robustbench_load_model
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.loaders import CustomImageFolder
from robustbench.data import PREPROCESSINGS
from torchvision.utils import save_image

import utils
from attacker import Attackers, attacker_list
from utils import overwrite_config, read_yaml, setup_logger
from utils.functions import experimental_post_process
from utils.Gather import GatherManager
from utils.common import get_machine_info, output_execute_info


def argparser():
    parser = ArgumentParser()

    parser.add_argument(
        "-o",
        "--result_dir",
        help="Result directory name (not path)",
        required=True,
        type=str,
    )
    parser.add_argument("-i", "--param", required=True, nargs="*", type=str)
    parser.add_argument(
        "--cmd_param",
        type=str,
        default=None,
        nargs="*",
        help='list of "param:cast type:value"'
        + " ex)  model_name:str:XXX solver.params.num_sample:int:10",
    )
    parser.add_argument("-g", "--gpu", type=int, default=None)
    parser.add_argument("--nth", "--num_thread", type=int, default=1)
    parser.add_argument(
        "--experiment",
        action="store_true",
    )
    parser.add_argument(
        "--export_statistics",
        action="store_true",
    )
    parser.add_argument(
        "--save_points",
        action="store_true",
    )
    parser.add_argument("--save_localopts", action="store_true")
    parser.add_argument("-bs", "--batch_size", type=int, default=1)
    parser.add_argument("--image_index", type=int, default=None)
    parser.add_argument(
        "--image_indices",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_image",
        action="store_true",
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=30,
        help="10:DEBUG,20:INFO,30:WARNING,40:ERROR,50:CRITICAL",
    )
    return parser


class ImageNetAttack(object):
    """ImageNet, Attackerclass.
    Parameters
    ----------
    result_dir : str

    save_image : bool
        true
    config: AttrDict
        attacker parameters

    Attributes
    ----------
    result_dir : str

    save_image : bool
        true
    config: AttrDict
        attacker parameters

    Notes
    -----
    https://robustbench.github.io/
    """

    x_test = None
    y_test = None
    model = None

    def __init__(self, result_dir=None, save_image=False, config=None) -> None:
        super().__init__()
        self.config = config
        self.result_dir = result_dir
        self.save_image = save_image

        # set device
        if config.gpu_id is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{config.gpu_id}")

        # load model
        logger.debug("\n loading model")
        self.load_model()

        # load dataset
        logger.debug("\n loading dataset")
        # self.load_dataset()
        self.model.to(self.device)

    def run(
        self,
        idx: int,
        bs: int = 1,
        EXP: bool = False,
        export: bool = False,
        image_indices=None,
        *args,
        **kwargs,
    ):
        """imagenet, Attack.

        Parameters
        ----------
        idx: int
            cifar10 index
        bs: int
            batchsize

        Returns
        -------
        adv_examples : dict
            key is label, value is image
        logs : dict
            key is label, value is log dictionary
        """

        # Gather Setting
        str_gather_name = f"{self.config.attacker}_{idx}_bs{bs}"
        gather = GatherManager().getGather(str_gather_name)
        staticsfilepath = (
            f"{self.result_dir}/{self.config.attacker}/{idx}/{gather.name}.pickle"
        )

        # setting attacker
        self.attacker = Attackers(
            self.config.attacker,
            model=self.model,
            W=None,
            bias=None,
            bound_opt=None,
            config=self.config,
            gather=gather,
            save_image=self.save_image,
            experiment=EXP,
            export_statistics=export,
            *args,
            **kwargs,
        )
        logger.info(self.attacker.__str__())
        if self.result_dir is not None:
            result_dir = f"{self.result_dir}/{self.attacker.name}/{idx}"
            os.makedirs(result_dir, exist_ok=True)
            setattr(self.config, "result_dir", result_dir)
        else:
            raise NotImplementedError

        execute_information_path = os.path.join(
            self.result_dir,
            self.attacker.name,
            "execute_information.json",
        )
        if not os.path.exists(execute_information_path):
            output_execute_info(config=config, save_path=execute_information_path)

        run_yaml_path = os.path.join(
            self.result_dir,
            self.attacker.name,
            "run.yaml",
        )
        if not os.path.exists(run_yaml_path):
            with open(run_yaml_path, "w") as file:
                yaml.dump(original_config, file)

        # load input image and label
        # batch
        batchBeginIdx = bs * idx
        batchEndIdx = bs * (idx + 1)

        # 
        attacked_images = torch.tensor(image_indices[batchBeginIdx:batchEndIdx]).type(
            torch.long
        )

        # =====

        image, label = self.load_dataset(attacked_images)
        # =====
        setattr(self.config, "attacked_images", attacked_images)

        pred = torch.nn.functional.softmax(
            self.model(image.to(self.device)), dim=1
        ).cpu()
        logs = {"info": {"idx": idx, "label": label, "pred": pred, "image": image}}
        adv_examples = {label: image}
        logger.debug(f'idx {idx} original: {label} {logs["info"]["pred"]}')

        # save original image
        if self.save_image and self.result_dir is not None:
            save_image_path = f"{result_dir}/original.png"
            save_image(image, save_image_path)

        if self.config.adv_target:
            # attack for each target class
            assert bs == 1
            gather.setParam("label", label)
            for target in range(10):
                if target == label:
                    continue
                adv_example, log = self.attacker.attack(
                    x=image,
                    label=label,
                    target=torch.LongTensor([target]),
                    log_prefix=f"{result_dir}/{target}",
                )

                adv_examples[target] = adv_example
                logs[target] = log
                if self.save_image and self.result_dir is not None:
                    save_image_path = f"{result_dir}/attacked_{target}.png"
                    save_image(adv_example, save_image_path)
        else:
            # try to misclassify the true class.
            adv_example, log = self.attacker.attack(x=image, y=label)
            if self.save_image:
                logger.info("\n[SAVE] vulnerable image")
                adv_examples["attacked"] = adv_example.cpu()

            logs["attacked"] = log
            if self.save_image and self.result_dir is not None:
                save_image_path = f"{result_dir}/attacked.png"
                save_image(adv_example, save_image_path)

        # save is adv_target or not.
        logs["adv_target"] = self.config.adv_target

        # save and display logs
        if self.result_dir is not None:
            save_log_path = f"{result_dir}/logs.pth"
            torch.save(
                logs,
                f=save_log_path,
            )
            # save config(yaml) file
            for yaml_path in self.config.yaml_paths:
                try:
                    shutil.copyfile(
                        yaml_path,
                        os.path.join(
                            self.config.result_dir, os.path.basename(yaml_path)
                        ),
                    )
                except shutil.SameFileError:
                    pass
            total_yaml_file = os.path.join(
                self.config.result_dir,
                "run.yml",
            )
            with open(total_yaml_file, "w") as yaml_file:
                yaml.dump(dict(config), yaml_file, default_flow_style=False)

        ## save_image
        if self.save_image:
            gather.setForceSave2pickle(staticsfilepath)

        # , itercsv.
        if EXP:
            bs = image.shape[0]
            from utils.functions import experimental_post_process

            experimental_post_process(
                config=self.config,
                gather=gather,
                num_restart=self.attacker.num_restarts,
                root=f"{self.result_dir}/{self.attacker.name}",
                index_start=batchBeginIdx,
                index_end=batchBeginIdx + bs,
                batch_idx=idx,
            )

        return adv_example, logs, save_log_path, staticsfilepath

    def load_model(self):
        """Load trained model instance."""
        model_name = self.config.model_name
        dataset = self.config.dataset
        threat_model = "Linf"

        logger.info(f"\n [MODEL] {model_name}. ")
        model = robustbench_load_model(
            model_name, dataset=dataset, threat_model=threat_model
        )
        self.prepr = all_models[BenchmarkDataset("imagenet")][
            ThreatModel(threat_model)
        ][model_name]["preprocessing"]

        self.model = model.to(self.device)

    def load_dataset(self, indices):
        """Load ImageNet data."""
        dataset_name = self.config.dataset
        assert dataset_name == "imagenet"
        assert self.config.n_examples <= 5000
        dataset = CustomImageFolder(
            "../storage/ILSVRC2012/ILSVRC2012_img_val_for_ImageFolder/val",
            transform=PREPROCESSINGS[self.prepr],
        )

        x_test = list()
        y_test = list()

        logger.debug(f"indices:{indices.tolist()}")

        for index in indices:
            x, y, _ = dataset.__getitem__(index)
            x_test.append(x.unsqueeze(0))
            y_test.append(y)

        x_test = torch.vstack(x_test)
        y_test = torch.tensor(y_test).type(torch.long)

        return x_test, y_test

    def log_gather(self, idx, x, label, logs, gather):
        logger.info(f'idx {idx} original: {label} {logs[label]["pred"]}')
        if self.config.adv_target:
            for target in range(10):
                if target == label:
                    continue

                target_score = logs[target]["pred"]
                outdict = {
                    "targetscore": target_score,
                }
                gather.updateDict(f"target_{target}", outdict)
        else:
            if "pred" in logs[label]:
                print(f'original: {label} {logs[label]["pred"]}')
            if "attacked" in logs.keys():
                print(f"attacked: {label} {logs['attacked']['pred']}")

    def log_display(self, idx, x, label, logs):
        logger.info(f'idx {idx} original: {label} {logs[label]["pred"]}')
        if self.config.adv_target:
            for target in range(10):
                if target == label:
                    continue
                logger.info(f'idx {idx} target: {target} {logs[target]["pred"]}')
        else:
            if "pred" in logs[label]:
                print(f'original: {label} {logs[label]["pred"]}')
            if "attacked" in logs.keys():
                print(f"attacked: {label} {logs['attacked']['pred']}")

    def __str__(self):
        s = f"Cifar10Attack:\n"
        s += f"    storage_dir: {self.config.storage_dir}\n"
        s += f"     result_dir: {self.result_dir}\n"
        s += f"       attacker: {self.config.attacker}\n"
        s += f"     model_path: {self.config.model_path}\n"
        s += f"     save_image: {self.save_image}"
        return s


def main(
    result_dir,
    save_image,
    config,
    idx,
    bs,
    EXP,
    export,
    save_points,
    save_localopts,
    image_indices,
):
    imagenet_attack = ImageNetAttack(
        result_dir=result_dir,
        save_image=save_image,
        config=config,
    )

    if type(idx) is str:
        idx = int(idx)
    adv_example, logs, save_log_path, staticsfilepath = imagenet_attack.run(
        idx,
        bs,
        EXP=EXP,
        export=export,
        save_points=save_points,
        save_localopts=save_localopts,
        image_indices=image_indices,
    )
    return adv_example, logs, save_log_path, staticsfilepath


if __name__ == "__main__":
    import warnings
    import multiprocessing as mp

    mp.set_start_method("spawn")
    warnings.simplefilter(action="ignore")

    parser = argparser()
    args = parser.parse_args()

    logger = setup_logger.setLevel(args.log_level)

    config = AttrDict(read_yaml(args.param))

    # overwrite by cmd_param
    if args.cmd_param is not None:
        config = overwrite_config(args.cmd_param, config)

    original_config = dict(config)

    # add other attributes.
    setattr(config, "gpu_id", args.gpu)
    setattr(config, "yaml_paths", args.param)

    #
    cmd_argv = " ".join((["python"] + sys.argv))
    config.cmd = cmd_argv

    #
    environ_variables = os.environ
    config.environ_variables = environ_variables

    # hash
    # try:
    #     git_hash = git.cmd.Git("./").rev_parse("HEAD")
    #     config.git_hash = git_hash

    #     #  branch
    #     _cmd = "git rev-parse --abbrev-ref HEAD"
    #     branch = subprocess.check_output(_cmd.split()).strip().decode("utf-8")
    #     branch = "-".join(branch.split("/"))
    #     config.branch = branch
    # except:
    #     pass

    #
    machine = get_machine_info()
    config.machine = machine

    # batch
    bs = args.batch_size
    assert bs > 0
    image_index = args.image_index

    # index
    image_indicies_yaml = args.image_indices

    torch.set_num_threads(args.nth)

    if image_index is not None:
        image_index = args.image_index
        logger.info(f"\n [Image Index] {image_index}")
        assert bs == 1
        main(
            result_dir=args.result_dir,
            save_image=args.save_image,
            config=config,
            idx=0,
            bs=bs,
            EXP=args.experiment,
            export=args.export_statistics,
            image_indices=[image_index],
        )
    else:
        image_indices = None
        if image_indicies_yaml is None:
            #
            image_indices = [i for i in range(config.n_examples)]
        else:
            #
            image_indices = read_yaml(image_indicies_yaml)["index"]

        num_batch = math.ceil(len(image_indices) / bs)

        for idx in range(num_batch):
            print("idx =", idx)
            main(
                result_dir=args.result_dir,
                save_image=args.save_image,
                config=config,
                idx=idx,
                bs=bs,
                EXP=args.experiment,
                export=args.export_statistics,
                image_indices=image_indices,
                save_points=args.save_points,
                save_localopts=args.save_localopts,
            )

else:
    logger = setup_logger(__name__)
