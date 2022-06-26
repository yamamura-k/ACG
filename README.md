# Diversified Adversarial Attacks based on Conjugate Gradient Method
This is the python implementation of our paper, "Diversified Adversarial Attacks based on Conjugate Gradient Method"
, accepted to ICML2022. [paper(arxiv)](https://arxiv.org/abs/2206.09628)

<br>
<br>

# Environment

<table border="5"  align="center">
<tr>
<td align="center">Python</td> <td colspan="3"  align="center">3.9.8</td>
</tr>
<td align="center">PyTorch</td><td colspan="3" align="center">1.10.0+cu113</td>
<tr>
<td align="center">gcc</td> 
<td align="center"> gcc version >= 5.4.0  </td>
</tr>
<tr>
<td align="center">CUDA</td> <td colspan="3" align="center"> 11.5</td>
</tr>
</table>

<br>
<br>

# Installation

+ Install python libraries.
```bash
pip install -r requirements.txt
```

+ Complie `.cpp` and `.c` codes.

```bash
cd src/utils/cluster_coef_c
python setup.py build_ext -i
```

+ Set ImageNet dataset.<br>
The directory name is the same as auto-attack

After your download, `ls` outputs the follow.
```bash
storage/ILSVRC2012/ILSVRC2012_img_val_for_ImageFolder/val
```

+ Downloads the robsust models from RobustBench.
```bash
cd src
python get_models.py
```


## Environment Variables

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

+ Fix `PYTHONHASHSEED` to 0.
```bash
export PYTHONHASHSEED=0
```
## Dataset

+ ImageNet
  1. `cd ../storage/ILSVRC2012`
  2. Download `ILSVRC2012_img_val.tar` and `ILSVRC2012_devkit_t12.tar.gz` from [ImageNet official site](https://image-net.org/index.php)
  ```bash
  $ ls
  ILSVRC2012_img_val.tar
  ```

  3. `mkdir val && tar -xf ILSVRC2012_img_val.tar -C ./val`

  4. `tar -xzf ILSVRC2012_devkit_t12.tar.gz`

  5. `python build_dataset.py`


## Usage
Attack on CIFAR-10
```bash
python -B run_cifar10_attack.py -o ../debug -g 0 --log_level 20 --param ./params/robustbench/cifar10/autoconjugate.yaml ./params/robustbench/cifar10/di.yaml  --experiment -bs 10
```
Attack on ImageNet
```bash
python -B run_imagenet_attack.py -o ../debug -g 0 --log_level 20 --param ./params/robustbench/imagenet/autoconjugate.yaml ./params/robustbench/cifar10/di.yaml  --experiment -bs 10
```
Attack on CIFAR-100
```bash
python -B run_cifar100_attack.py  -o ../debug -g 0 --log_level 20 --param ./params/robustbench/cifar100/autoconjugate.yaml ./params/robustbench/cifar10/di.yaml  --experiment -bs 10
```


## Calculate the attack success rates from result dir.

```bash
(find ../result/ -maxdepth 7 |grep -e AUTOP -e AUTOC | xargs -L1 -P1 python run_evaluator_from_csv.py -ns 1  -r && find ../result/ -maxdepth 7 |grep AUTOC | xargs -L1 -P1 python run_evaluator_from_csv.py -ns 5 -r ) > cifar10_cw_result.csv
```


and open `cifar10_cw_result.csv`

```csv
Ding2020MMA,AUTOConjugaterestart-1,WideResNet-28-4,\cite{Ding2019},53.40,Wed Jan 26 10:59:21 2022
Ding2020MMA,AUTOConjugate,WideResNet-28-4,\cite{Ding2019},55.77,Wed Jan 26 10:59:21 2022
```
+ 1st column: model name listed in RobustBench.
+ 2nd column: the algorithm name. <br>
`AUTOConjugaterestart-1` mean ACG with one restart.
+ 3rd column: the architecture of model
+ 4th column: the citation of the adversarial training method
+ 5th column: the attack success rates
+ 6th column: the execution start time


## Docker Usage
### Requirements
- Docker \
  https://www.docker.com/
- NVIDIA Container Toolkit \
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html
### Command

Build Docker Command
```bash
docker build --rm -t autocg:latest .
```

Create docker instance
```bash
docker run -it --gpus all -v $PWD/src:/AutoCG/src autocg /bin/bash
```

Start created container instance
```bash
docker start -ai [ContainerID]
```

Connect started container
```bash
docker attach [ContainerID]
```

Detach from Container
```
[control-P] [control-Q]
```

## Citation
```
@inproceedings{yamamura2022,
    title={Diversified Adversarial Attacks based on Conjugate Gradient Method}, 
    author={Keiichiro Yamamura and Haruki Sato and Nariaki Tateiwa and Nozomi Hata and Toru Mitsutake and Issa Oe and Hiroki Ishikura and Katsuki Fujisawa},
    booktitle={ICML},
    year={2022}
}
```
