import tqdm
from robustbench import load_model

if __name__ == "__main__":
    # imagenet
    models = [
        "Salman2020Do_50_2",
        "Salman2020Do_R50",
        "Engstrom2019Robustness",
        "Wong2020Fast",
        "Salman2020Do_R18",
    ]
    for model in tqdm.tqdm(models):
        try:
            load_model(model, dataset="imagenet", threat_model="Linf")
        except:
            print(f"skip: {model}")

    # cifar10
    models = [
        "Andriushchenko2020Understanding",
        "Carmon2019Unlabeled",
        "Sehwag2020Hydra",
        "Wang2020Improving",
        "Hendrycks2019Using",
        "Rice2020Overfitting",
        "Zhang2019Theoretically",
        "Engstrom2019Robustness",
        "Chen2020Adversarial",
        "Huang2020Self",
        "Pang2020Boosting",
        "Wong2020Fast",
        "Ding2020MMA",
        "Zhang2019You",
        "Zhang2020Attacks",
        "Wu2020Adversarial_extra",
        "Wu2020Adversarial",
        "Gowal2020Uncovering_70_16",
        "Gowal2020Uncovering_70_16_extra",
        "Gowal2020Uncovering_34_20",
        "Gowal2020Uncovering_28_10_extra",
        "Sehwag2021Proxy",
        "Sehwag2021Proxy_R18",
        "Sitawarin2020Improving",
        "Chen2020Efficient",
        "Cui2020Learnable_34_20",
        "Cui2020Learnable_34_10",
        "Zhang2020Geometry",
        "Rebuffi2021Fixing_28_10_cutmix_ddpm",
        "Rebuffi2021Fixing_106_16_cutmix_ddpm",
        "Rebuffi2021Fixing_70_16_cutmix_ddpm",
        "Rebuffi2021Fixing_70_16_cutmix_extra",
        "Sridhar2021Robust",
        "Sridhar2021Robust_34_15",
        "Rebuffi2021Fixing_R18_ddpm",
        "Rade2021Helper_R18_extra",
        "Rade2021Helper_R18_ddpm",
        "Rade2021Helper_extra",
        "Rade2021Helper_ddpm",
        "Huang2021Exploring",
        "Huang2021Exploring_ema",
        "Gowal2021Improving_70_16_ddpm_100m",
    ]
    for model in tqdm.tqdm(models):
        try:
            load_model(model, dataset="cifar10", threat_model="Linf")
        except:
            print(f"skip: {model}")

    # cifar100
    models = [
        "Rice2020Overfitting",
        "Rebuffi2021Fixing_70_16_cutmix_ddpm",
        "Rebuffi2021Fixing_28_10_cutmix_ddpm",
        "Wu2020Adversarial",
        "Cui2020Learnable_34_10_LBGAT6",
        "Addepalli2021Towards_WRN34",
        "Addepalli2021Towards_PARN18",
        "Hendrycks2019Using",
        "Gowal2020Uncovering",
        "Cui2020Learnable_34_20_LBGAT6",
        "Cui2020Learnable_34_10_LBGAT0",
        "Sitawarin2020Improving",
        "Rebuffi2021Fixing_R18_ddpm",
        "Chen2020Efficient",
        "Gowal2020Uncovering_extra",
        "Chen2021LTD_WRN34_10",
        "Rade2021Helper_R18_ddpm",
    ]
    for model in tqdm.tqdm(models):
        try:
            load_model(model, dataset="cifar100", threat_model="Linf")
        except:
            print(f"skip: {model}")
