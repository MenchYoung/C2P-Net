# C2P-Net
This repository contains the official implementation of the paper:
“Contrastive Cross-Attention for Clinically-Consistent Survival Prediction from Longitudinal CT Scans” (Submitted to EMBC 2026)

## 🚀Introduction

Accurate prediction of Progression-Free Survival (PFS) from longitudinal computed tomography (CT) scans is crucial for personalizing cancer treatment planning. However, existing methods often fail to effectively model temporal tumor dynamics and can suffer from conflicting objectives in multi-task settings. To address these challenges, we propose the Contrastive Cross-Attention for Clinically-Consistent Prediction Network (C2P-Net). C2P-Net integrates an Iterative Cross-Attention (ICA) module for temporal fusion, a Contrastive Cross-Attention (CCA) mechanism to capture pathological changes, and a clinical logic consistency constraint to align multi-task objectives.
On a longitudinal CT cohort of 864 lung cancer patients, C2P-Net predicts progression status at multiple clinically relevant time points and achieves a C-index of 0.676; moreover, replacing standard cross-attention with the proposed CCA yields an additional 1.3\% AUC gain at the 6-month prediction task. Furthermore, the proposed model enables effective risk stratification, resulting in high- and low-risk groups with a statistically significant difference in PFS (P < 0.001).


![Network Architecture](fig.png)



## 🛠️ Installation

```bash
git clone https://github.com/MenchYoung/C2P-Net.git
cd C2P-Net
pip install -r requirements.txt
```

## 🏃‍♂️Training
```bash
python main.py
```

## 🏃‍♂️test
```bash
python test.py
```

## 📝 Citation
If you find this work useful, please cite our paper: comming soon...
