# Explainable AI for Alzheimer's Disease Stage Classification from MRI

**Team 09** · Explainable Artificial Intelligence Course

---

## Team Members

| Name | Notebooks |
|---|---|
| Omar Zayed | ConvMixer + RF, ResNet-50, SE-CNN + RF |
| Farida Ali | AlexNet + SVM, CNN, DAD-Net |
| Malak Khaled | VGG19 + SVM (Patient-Level), VGG19 + SVM (Scan-Level), Dual CNN, DenseNet Multiclass |
| Ahmed Moatasem | EfficientNet-B0 (K-Fold), EfficientNet-B0 (Multiclass), HTLML Hybrid, ResNet-18 + DenseNet-121 Hybrid, VGG16 + SVM (Binary), Kaggle→ADNI Transfer Learning, Cross-Dataset Validation |

---

## Dataset

All notebooks run on **Kaggle** and expect the **ADNI** dataset to be available.

**Step 1 — Add the public Kaggle dataset**

In the right-hand Kaggle panel click **Add Data** and search for:
```
aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented
```

**Step 2 — Add the ADNI dataset**

Upload your `adni_nifti_preprocessed` folder to Kaggle as a **Private Dataset** (Kaggle → Datasets → New Dataset → keep it Private), then add it to the notebook via **Add Data → Your Datasets**.

Alternatively, every notebook contains a setup cell that downloads the ADNI data automatically from Google Drive using `gdown` — no manual upload needed if internet is ON in the Kaggle session.

---

## Requirements

All notebooks self-install their dependencies in the first cell. No local setup is needed.

If you want to run locally, install with:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install grad-cam lime shap captum scikit-learn xgboost timm tqdm seaborn matplotlib pillow opencv-python
```

> Python 3.9+ recommended. GPU strongly recommended (CUDA 11.8+).

---

## Notebooks

Each notebook is fully self-contained: data loading → preprocessing → model training → evaluation → ≥ 4 XAI techniques.

### Omar Zayed

| Notebook | Model | XAI Techniques |
|---|---|---|
| `Team_09_ConvMixer_RF_Omar_Zayed.ipynb` | ConvMixer backbone + Random Forest | Grad-CAM, LIME, SHAP (TreeExplainer), Occlusion Sensitivity |
| `Team_09_ResNet50_Omar_Zayed.ipynb` | ResNet-50 (K-Fold, 1-vs-All) | Grad-CAM, LIME, SHAP, Integrated Gradients |
| `Team_09_SE_CNN_RF_Omar_Zayed.ipynb` | SE-CNN + Random Forest | Grad-CAM, LIME, SHAP, Feature Importance |

### Farida Ali

| Notebook | Model | XAI Techniques |
|---|---|---|
| `Team_09_AlexNet_SVM_Farida_Ali.ipynb` | AlexNet + SVM | Grad-CAM, LIME, SHAP, Occlusion Sensitivity |
| `Team_09_CNN_Farida_Ali.ipynb` | Custom CNN | Grad-CAM, LIME, SHAP, Occlusion Sensitivity, Integrated Gradients, Captum |
| `Team_09_DAD_Net_Farida_Ali.ipynb` | DAD-Net (Ahmed et al., 2022) | Grad-CAM, LIME, Integrated Gradients, Saliency Maps, SmoothGrad, Captum |

### Malak Khaled

| Notebook | Model | XAI Techniques |
|---|---|---|
| `Team_09_VGG19_SVM_PatientLevel_Malak_Khaled.ipynb` | VGG-19 + SVM (patient-level) | Grad-CAM, LIME, SHAP, Integrated Gradients, Saliency Maps |
| `Team_09_VGG19_SVM_ScanLevel_Malak_Khaled.ipynb` | VGG-19 + SVM (scan-level) | Grad-CAM, LIME, SHAP, Saliency Maps, Captum |
| `Team_09_DualCNN_Malak_Khaled.ipynb` | Dual-branch CNN | Grad-CAM, LIME, Occlusion Sensitivity, Saliency Maps |
| `Team_09_DenseNet_Multiclass_Malak_Khaled.ipynb` | DenseNet-121 Multiclass | Grad-CAM, LIME, SHAP, Integrated Gradients, Saliency Maps, Captum |

### Ahmed Moatasem

| Notebook | Model | XAI Techniques |
|---|---|---|
| `Team_09_EfficientNetB0_KFold_Ahmed_Moatasem.ipynb` | EfficientNet-B0 (patient K-Fold, 1-vs-All) | Grad-CAM, LIME, SHAP, Integrated Gradients |
| `Team_09_EfficientNetB0_Multiclass_Ahmed_Moatasem.ipynb` | EfficientNet-B0 Multiclass | Grad-CAM, LIME, SHAP, Integrated Gradients, Captum |
| `Team_09_HTLML_Hybrid_Ahmed_Moatasem.ipynb` | HTLML Hybrid (InceptionV3 + VGG16 → SVM/NB/XGB) | Grad-CAM, LIME, SHAP, Integrated Gradients, Captum |
| `Team_09_ResNet18_DenseNet121_Hybrid_Ahmed_Moatasem.ipynb` | ResNet-18 + DenseNet-121 Hybrid | Grad-CAM, LIME, SHAP, Integrated Gradients, Captum |
| `Team_09_VGG16_SVM_Binary_Ahmed_Moatasem.ipynb` | VGG-16 + SVM (3 binary tasks) | Grad-CAM, LIME, SHAP (KernelExplainer), Permutation Importance |
| `Team_09_Kaggle_ADNI_Transfer_Learning_Ahmed_Moatasem.ipynb` | ResNet-50 (Kaggle pre-train → ADNI fine-tune) | Grad-CAM, LIME, SHAP, Integrated Gradients |
| `Team_09_Cross_Dataset_Validation_Ahmed_Moatasem.ipynb` | ResNet-50 (zero-shot Kaggle → ADNI) | Grad-CAM, LIME, SHAP, Integrated Gradients |

---

## How to Run

1. Open any notebook on **Kaggle**.
2. Enable **GPU** (Settings → Accelerator → GPU T4 × 2 or P100).
3. Enable **Internet** (Settings → Internet → On) — required for `gdown` and `pip install`.
4. Add the two datasets as described in the [Dataset](#dataset) section above.
5. Click **Run All**.

Each notebook is independent — you can run any one on its own.

---

## Classification Classes

| Label | Meaning |
|---|---|
| `CN` | Cognitively Normal |
| `MCI` | Mild Cognitive Impairment |
| `AD` | Alzheimer's Disease |

---

## Repository Structure

```
├── Team_09_ConvMixer_RF_Omar_Zayed.ipynb
├── Team_09_ResNet50_Omar_Zayed.ipynb
├── Team_09_SE_CNN_RF_Omar_Zayed.ipynb
├── Team_09_AlexNet_SVM_Farida_Ali.ipynb
├── Team_09_CNN_Farida_Ali.ipynb
├── Team_09_DAD_Net_Farida_Ali.ipynb
├── Team_09_VGG19_SVM_PatientLevel_Malak_Khaled.ipynb
├── Team_09_VGG19_SVM_ScanLevel_Malak_Khaled.ipynb
├── Team_09_DualCNN_Malak_Khaled.ipynb
├── Team_09_DenseNet_Multiclass_Malak_Khaled.ipynb
├── Team_09_EfficientNetB0_KFold_Ahmed_Moatasem.ipynb
├── Team_09_EfficientNetB0_Multiclass_Ahmed_Moatasem.ipynb
├── Team_09_HTLML_Hybrid_Ahmed_Moatasem.ipynb
├── Team_09_ResNet18_DenseNet121_Hybrid_Ahmed_Moatasem.ipynb
├── Team_09_VGG16_SVM_Binary_Ahmed_Moatasem.ipynb
├── Team_09_Kaggle_ADNI_Transfer_Learning_Ahmed_Moatasem.ipynb
├── Team_09_Cross_Dataset_Validation_Ahmed_Moatasem.ipynb
├── data/
│   └── dataset_images/
├── team09_preprocessing.ipynb
├── eda-adni.ipynb
└── README.md
```

---

## License

MIT
