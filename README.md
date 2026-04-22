# early-alzheimer-detection
#  Multimodal Alzheimer’s Disease Classification

### Using DTI Imaging, APOE Genetic Features, and Clinical Data

---

##  Overview

This project focuses on the classification of Alzheimer’s Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) subjects using a **multimodal deep learning approach**.

The model combines:

*  **DTI brain images** (computer vision)
*  **APOE genetic feature**
*  **Clinical data (ADNIMERGE mapping)**

A custom dataset was created by mapping imaging data with clinical and genetic features using patient identifiers (RID).

---

##  Objectives

* Classify subjects into **CN, MCI, AD**
* Use **DTI-based CNN** for image feature extraction
* Integrate **APOE + clinical data** in a multimodal framework
* Ensure **patient-level data splitting** to avoid leakage

---

##  Project Structure

```
├── dataset_creation_colab.ipynb     # Dataset construction (Colab)
├── training_kaggle.ipynb           # Model training & evaluation (Kaggle)
├── processed_images/               # DTI slices (input images)
├── README.md
```

---

##  Workflow

###  Step 1: Dataset Creation (Google Colab)

File: `dataset_creation_colab.ipynb`

* Extract DTI image slices
* Parse filenames to obtain **RID (patient ID)**
* Map each image with:

  * Label (CN / MCI / AD)
  * APOE feature
* Filter valid samples:

```python
if rid in label_map and rid in apoe_map:
```

* Normalize and resize images to **128×128**
* Save structured dataset

---

###  Step 2: Model Training (Kaggle)

File: `training_kaggle.ipynb`

* Load processed dataset
* Apply **patient-level splitting (GroupShuffleSplit)**
* Build multimodal model:

  * CNN for DTI images
  * Dense layer for APOE
  * Feature fusion (concatenation)
* Train model with:

  * EarlyStopping
  * ReduceLROnPlateau
* Evaluate using:

  * Accuracy
  * Confusion Matrix
  * Classification Report

---

##  Model Architecture

```
DTI Image → CNN → Feature Vector (32D)
                     ↓
APOE → Dense → Feature (1D)

→ Concatenate → Dense → Softmax → Output (CN / MCI / AD)
```
---

### Key Observations:

* Image-based CNN performed most reliably
* MCI class was difficult to learn
* Multimodal fusion needs better feature balancing
* Proper data splitting significantly impacts results

---

## Key Challenges

* **Class imbalance (MCI underrepresented)**
* **Feature imbalance (image vs APOE)**
* **Small dataset size**
* **High variance across runs**

---

##  Novel Contributions

*  Custom multimodal dataset creation (DTI + APOE + clinical)
*  Lightweight CNN architecture
*  Experimental analysis of multiple training strategies

---

##  Limitations

The performance is influenced by dataset size and class imbalance, which are common challenges in medical imaging-based deep learning applications.

---

##  Future Work

* Incorporate more clinical features
* Use richer genetic representations
* Apply feature scaling/attention for multimodal fusion
* Increase dataset size (more subjects from ADNI)
* Explore 3D CNN or volumetric approaches

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy / Pandas
* Scikit-learn

---

##  Notes

* Dataset creation and training are separated for modularity
* Designed for reproducibility and experimentation
* Focused on **Computer Vision + Multimodal Learning**

---

##  Acknowledgment

* ADNI Dataset
* Research papers on DTI-based Alzheimer classification

---
