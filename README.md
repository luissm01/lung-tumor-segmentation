# Lung Tumor Segmentation with U-Net

This project implements a full deep learning pipeline for segmenting lung tumors in CT scans using a U-Net architecture. It includes preprocessing, data handling, training, evaluation, and visualization.

---

## Dataset

The dataset comes from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) (Task06_Lung). It includes:
- CT volumes (`imagesTr/`, `imagesTs/`)
- Segmentation masks (`labelsTr/`)

All data is stored in the `datos/` folder:
```
datos/
├── imagesTr/           # Raw training CT volumes
├── labelsTr/           # Corresponding segmentation masks
├── imagesTs/           # Test CT volumes without labels
└── Preprocessed/       # Normalized 2D slices in .npy format (train/val/test)
```

> Data license: [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/luissm01/lung-tumor-segmentation.git
cd lung-tumor-segmentation
```
### 3. Install dependencies
```bash
conda env create -f environment.yml
conda activate lung-tumor-segmentation
```
### OR
### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Pipeline Overview

### Preprocessing
- Applies lung windowing (`level=-600`, `width=1500`)
- Crops irrelevant anatomy (removes 30 initial axial slices)
- Normalizes to [0, 1] and resizes to 256×256
- Saves each slice and mask as `.npy` files

### Dataset
- Custom PyTorch `Dataset` class
- Uses `imgaug` for data augmentation
- Applies weighted random sampling to address tumor rarity

### Model
- Custom 2D U-Net with:
  - BatchNorm and Dropout
  - Xavier weight initialization
  - Logit output (no sigmoid)

### Training
- PyTorch Lightning module
- Dice + Focal Loss (`monai`)
- Early stopping + checkpointing
- TensorBoard logging

---

## Training the Model

**Run from script (recommended due to multiprocessing issues):**
```bash
python notebooks/train.py
```

---

## Visual Test Results

Test predictions are run slice-by-slice over entire CT volumes. For each patient, we generate a video that overlays the predicted tumor regions.

📹 [Example test segmentation video](https://github.com/luissm01/lung-tumor-segmentation/tree/main/images/test_videos)

Videos are saved in the `images/` folder.

---

## Performance

- Validation Dice ranged from **0.15 to 0.25**
- Tumor detection stabilized around **epoch 17–18**
- Training Dice and loss plateaued early
- Visual inspection confirms the model identifies tumors when present

---

## Final Thoughts

Training a model to detect lung tumors in CT scans is a difficult task:
- Tumors are **small and rare**
- The dataset is **limited**
- We work with **2D slices only**, without volumetric context

That said, the model learned to localize tumors reasonably well. It produces consistent predictions, and often correctly highlights the tumor when it is visible. With more data and more time, performance could certainly improve—but this version already provides a **realistic and interpretable baseline**.

---

## Next Steps

- Incorporate 3D context (e.g., 3D U-Net)
- Use semi-supervised methods to leverage test data
- Apply postprocessing (e.g., remove small blobs)
- Build a web viewer (e.g., Streamlit) for interactive exploration

---

## License

This project uses public data under the [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

---

## About the Author

This project was developed by **Luis Sánchez Moreno**, Biomedical Engineer specialized in health data analysis and deep learning.

It was part of a personal learning journey focused on applying AI to real medical challenges. The entire pipeline was built from scratch, including data preprocessing, model training, evaluation, and visualization.

I'm passionate about building interpretable, clinically useful models—even when working with limited data and resources.  
If you're interested in collaboration or just want to share feedback, feel free to reach out or connect on [LinkedIn](https://www.linkedin.com/in/tu-linkedin) or [GitHub](https://github.com/luissm01).
