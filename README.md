# C2P-Net
This repository contains the official implementation of the paper:
“Contrastive Cross-Attention for Clinically-Consistent Survival Prediction from Longitudinal CT Scans” (Submitted to EMBC 2026)

## 🚀Introduction

Accurate prediction of Progression-Free Survival (PFS) from longitudinal computed tomography (CT) scans is crucial for personalizing cancer treatment planning. However, existing methods often fail to effectively model temporal tumor dynamics. To address this challenge, we propose the Contrastive Cross-Attention Network with Clinical Consistency Constraint (C2P-Net), which jointly predicts the PFS risk score and progression probabilities at multiple time points. C2P-Net integrates an Iterative Cross-Attention (ICA) module for temporal information fusion, a Contrastive Cross-Attention (CCA) mechanism to highlight features related to pathological changes and a Clinical Consistency Constraint to align the multi-task objectives. In a longitudinal CT cohort of 864 lung cancer patients, C2P-Net achieves an AUC of 0.810 for 6-month progression prediction and a C-index of 0.676 for PFS prediction, demonstrating robust performance across prediction tasks at multiple follow-up time points. Furthermore, the proposed model enables clear risk stratification, resulting in high-risk and low-risk groups with a statistically significant difference in PFS with a log-rank p-value less than 0.001. Overall, these results highlight the potential of C2P-Net to enhance PFS prediction and time-specific progression assessment from longitudinal CT data.


![Network Architecture](structure.png)



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
