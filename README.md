# Fine-Grained Vehicle Recognition using Integrated Attention Modules

## Overview

This repository contains the full implementation and experimental results for a comparative study of attention-augmented convolutional neural networks applied to fine-grained vehicle recognition. Three architectures are evaluated:

- **Standard ResNet-50** — ImageNet-pretrained baseline
- **SE-ResNet-50** — Squeeze-and-Excitation channel attention (Hu et al., 2018)
- **CBAM-ResNet-50** — Convolutional Block Attention Module, channel + spatial attention (Woo et al., 2018)

Experiments are conducted on two benchmark datasets - Stanford Cars and CompCars - under two evaluation protocols: cross-validated classification on the training partition (**Phase 1**), and test-set evaluation comparing ground-truth bounding box crops against automatic YOLO-based detections (**Phase 2**).

## Repository Structure

VehicleRecognition/
├── experiments/
│ ├── models.py # SE-ResNet-50, CBAM-ResNet-50, Standard ResNet-50
│ ├── datasets.py # Dataset class, train/val augmentation pipeline
│ ├── pretests_phase1.py # Coordinate-descent hyperparameter search
│ ├── ex_phase1.py # Phase 1: RSKF cross-validation experiment
│ ├── ex_phase2.py # Phase 2: GT bbox vs YOLO crop experiment
│ ├── figures/ # Publication-quality figures (PDF + PNG)
│ ├── example gradcams/ # GradCAM visualisations per configuration
│ ├── results_standard.json # Phase 1 fold results – Standard ResNet-50
│ ├── results_se.json # Phase 1 fold results – SE-ResNet-50
│ ├── results_cbam.json # Phase 1 fold results – CBAM-ResNet-50
│ ├── phase1_analysis.txt # Full statistical analysis output
│ ├── phase2_state_stanford.json # Phase 2 fold results – Stanford
│ ├── phase2_state_compcars.json # Phase 2 fold results – CompCars
│ ├── phase2_results_stanford.txt
│ ├── phase2_results_compcars.txt
│ ├── pretest_data_stanford.json # Hyperparameter search results
│ ├── yolo_cache_stanford.json # Cached YOLO bounding boxes
│ └── yolo_cache_compcars.json
├── StanfordCars_devkit/
│ ├── cars_train.csv
│ ├── cars_test.csv
│ └── convert_stanford.py # Annotation converter (require the withlabels .mat file, which is included in this project, sourced from here: https://github.com/jhpohovey/StanfordCars-Dataset)
├── CompCars_devkit/
│ ├── compcars_dataset.csv
│ └── convert_compcars.py # Annotation converter
├── data/ # Dataset images — not tracked by git
│ ├── StanfordCars/
│ └── CompCars/
└── .gitignore

## Datasets

Both datasets require a manual download; the `data/` directory is excluded from version control.

### Stanford Cars

- **196 classes** (make, model, year)
- 8,144 training images / 8,041 test images
- Annotations: class-labelled bounding boxes (`.mat` devkit)
- Source: https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset
- After downloading, place under `data/StanfordCars/` and run `StanfordCars_devkit/convert_stanford.py` to produce `cars_train.csv` and `cars_test.csv`. The test annotation file `SC_cars_test_annos_withlabels.mat` is required for the test CSV.

### CompCars

- **431 classes** (make–model, using the official classification split)
- Images organised by model ID and year (2009–2015)
- Source: https://www.kaggle.com/datasets/renancostaalencar/compcars
- After downloading, place under `data/CompCars/` and run `CompCars_devkit/convert_compcars.py`.

---

### Hyperparameter Search (pretests)

A coordinate-descent search over four parameters is run prior to the main experiments using a single training/validation split on Stanford Cars. Parameters searched and their candidate values:

| Parameter      | Candidates              | Selected |
| -------------- | ----------------------- | -------- |
| Epochs         | 10, 20, 30, **50**      | 50       |
| Batch size     | **16**, 32, 64          | 16       |
| Warmup epochs  | 0, 3, **5**, 10         | 5        |
| LR (new heads) | 0.01, **0.001**, 0.0001 | 0.001    |

Fixed throughout: backbone LR = 1×10⁻⁵, weight decay = 1×10⁻⁴.

Run: `python pretests_phase1.py --dataset stanford`

### Phase 1 — Cross-Validated Classification

Repeated Stratified K-Fold (5 folds × 2 repeats = **10 iterations**) on the training partition. All three models are evaluated per fold; after all model runs complete, a Friedman test is applied across the 10 fold scores followed by Wilcoxon signed-rank post-hoc tests with Bonferroni correction (α = 0.0167).

Each model run is independent and resumes automatically from any previously completed folds:

```bash
python ex_phase1.py --model standard
python ex_phase1.py --model se
python ex_phase1.py --model cbam
```

Any of the three commands regenerates all figures once all results are on disk.

### Phase 2 — GT Bounding Box vs YOLO Detection

Models are trained on the full training partition (same RSKF splits as Phase 1) and evaluated on the official test set under two conditions:

- **Experiment A (GT):** images cropped using ground-truth bounding boxes from the dataset annotations
- **Experiment B (YOLO):** images cropped using the highest-confidence vehicle detection from a YOLO detector (COCO classes: car, bus, truck); falls back to the full image when no vehicle is detected

YOLO bounding boxes are cached to JSON after the first run to avoid redundant inference.

```bash
python ex_phase2.py --dataset stanford
python ex_phase2.py --dataset compcars
```

## Installation

**Requirements:** Python 3.12, CUDA-capable GPU recommended.

```bash
pip install torch torchvision
pip install scikit-learn scipy pandas pillow
pip install ultralytics matplotlib
```

Remember to change the paths to data and data devkits.

The YOLO model weights (`yolo26s.pt`) are downloaded automatically by Ultralytics on first use and are excluded from version control.

## Main References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. _CVPR 2016._
2. Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. _CVPR 2018._
3. Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. _ECCV 2018._
4. Krause, J., Stark, M., Deng, J., & Fei-Fei, L. (2013). 3D Object Representations for Fine-Grained Categorization. _ICCV Workshops 2013._ (Stanford Cars dataset)
5. Yang, L., Luo, P., Loy, C. C., & Tang, X. (2015). A Large-Scale Car Dataset for Fine-Grained Categorization and Verification. _CVPR 2015._ (CompCars dataset)
