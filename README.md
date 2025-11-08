# AdaSCALE: Adaptive Scaling for OOD Detection 
This codebase provides a Pytorch implementation of:

>**AdaSCALE: Adaptive Scaling for OOD Detection**  
[![AdaSCALE](https://img.shields.io/badge/arXiv'25-AdaSCALE-fdd7e6?style=for-the-badge)](https://arxiv.org/abs/2503.08023)  
Sudarshan Regmi

>**AdaSCALE's illustration**
<p align="center">
  <img width="1000" src="imgs/adascale.png">
</p>

>**Comparison with fixed scaling procedure**
<p align="center">
  <img width="400" src="imgs/teaser.png">
</p>

## Abstract
The ability of the deep learning model to recognize when a sample falls outside its learned distribution is critical for safe and reliable deployment. Recent state-of-the-art out-of-distribution (OOD) detection methods leverage activation shaping to improve the separation between in-distribution (ID) and OOD inputs. These approaches resort to sample-specific scaling but apply a static percentile threshold across all samples regardless of their nature, resulting in suboptimal ID-OOD separability. In this work, we propose \textbf{AdaSCALE}, an adaptive scaling procedure that dynamically adjusts the percentile threshold based on a sample's estimated OOD likelihood. This estimation leverages our key observation: OOD samples exhibit significantly more pronounced activation shifts at high-magnitude activations under minor perturbation compared to ID samples. AdaSCALE enables stronger scaling for likely ID samples and weaker scaling for likely OOD samples, yielding highly separable energy scores. Our approach achieves state-of-the-art OOD detection performance, outperforming the latest rival OptFS by 14.94% in near-OOD and 21.67% in far-OOD datasets in average FPR@95 metric on the ImageNet-1k benchmark across eight diverse architectures. 

<span>Check other works:</span>

<a href="https://github.com/sudarshanregmi/t2fnorm"><img src="https://img.shields.io/badge/CVPRW'24-T2FNorm-fdd7e6?style=for-the-badge" alt="t2fnorm" style="margin-right: 10px;"></a> <br>
<a href="https://github.com/sudarshanregmi/reweightood"><img src="https://img.shields.io/badge/CVPRW'24-ReweightOOD-f4d5b3?style=for-the-badge" alt="reweightood" style="margin-right: 10px;"></a> <br>
<a href="https://github.com/sudarshanregmi/ascood"><img src="https://img.shields.io/badge/arXiv'25-ASCOOD-fdd7e6?style=for-the-badge" alt="ascood" style="margin-right: 10px;"></a>

## Note
You only need to tune `p_max`. Keep the remaining hyperparameters fixed to the following defaults for near‑optimal performance:
- lambda = 10
- p_min = 60
- o = 5% (0.05)
- k1 = 5% (0.05)
- k2 = 1% (0.01)
- epsilon = 0.5

Erratum: There is a minor paper/code mismatch. In the code, `k1` and `k2` are swapped relative to the paper:
- paper `k1` ↔ code `k2`
- paper `k2` ↔ code `k1`

### Follow [OpenOOD](https://github.com/Jingkang50/OpenOOD) official instruction to complete the setup.
```
pip install git+https://github.com/Jingkang50/OpenOOD
```

### Evaluation setup
<p align="center">
  <img width="500" src="imgs/eval.png">
</p>

### Example Scripts
Use the following scripts for inferencing with AdaSCALE-A postprocessor on different datasets:

- **CIFAR-10:**
  ```bash
  bash scripts/ood/adascale_a/cifar10_train_adascale.sh
  bash scripts/ood/adascale_a/cifar10_test_adascale.sh
  ```
- **CIFAR-100:**
  ```bash
  bash scripts/ood/adascale_a/cifar100_train_adascale.sh
  bash scripts/ood/adascale_a/cifar100_test_adascale.sh
  ```
- **ImageNet-200:**
  ```bash
  bash scripts/ood/adascale_a/imagenet200_train_adascale.sh
  bash scripts/ood/adascale_a/imagenet200_test_adascale.sh
  ```
- **ImageNet-1k:**
  ```bash
  bash scripts/ood/adascale_a/imagenet_train_adascale.sh
  bash scripts/ood/adascale_a/imagenet_test_adascale.sh
  ```

Please see [**./scripts/ood/adascale_l/**](./scripts/ood/adascale_l/)  folder for another variant.

Please see [**results folder**](./results/) for [**OpenOOD v1.5 benchmark**](https://zjysteven.github.io/OpenOOD/#leaderboard) results.

Please refer to [**Google Drive**](https://drive.google.com/drive/folders/1bSymS3257nPsP83xCQGISYc1NBrdTgc3?usp=sharing) for access to the models we trained on CIFAR-10/100 datasets.

### Results
- AdaSCALE's generalization in ImageNet-1k benchmark:
<p align="center">
  <img width="1000" src="imgs/imagenet.png">
</p>

- AdaSCALE's compatibility with ISH
<p align="center">
  <img width="500" src="imgs/ish.png">
</p>

- AdaSCALE's competitiveness in CIFAR benchmarks:
<p align="center">
  <img width="500" src="imgs/cifar.png">
</p>

- Adaptive percentile vs. Static percentile
<p align="center">
  <img width="500" src="imgs/percentile.png">
</p>

- AdaSCALE's efficacy with limited ID data
<p align="center">
  <img width="500" src="imgs/limited_id_data.png">
</p>

### Consider citing this work if you find it useful.

```
@misc{regmi2025adascaleadaptivescalingood,
      title={AdaSCALE: Adaptive Scaling for OOD Detection},
      author={Sudarshan Regmi},
      year={2025},
      eprint={2503.08023},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.08023},
}
```
### Acknowledgment
This codebase builds upon [OpenOOD](https://github.com/Jingkang50/OpenOOD).
