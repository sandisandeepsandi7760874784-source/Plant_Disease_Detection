# 🌿 Deep Learning-Based Plant Disease Detection

An advanced, end-to-end computer vision and deep learning project implemented entirely inside **Jupyter Notebook** (`Plant_Disease_Detection.ipynb`) using Python and **TensorFlow / Keras MobileNetV2 Transfer Learning**.

The project accurately detects and diagnoses foliar diseases across **Tomato** and **Potato** crops from the benchmark **PlantVillage** dataset, providing confidence probabilities and actionable agricultural remedies.

---

## 📁 Repository Structure

```text
d:\Plant Disease Detection\
│
├── Plant_Disease_Detection.ipynb    # Main end-to-end Jupyter Notebook (fully pre-executed with plots & metrics)
├── build_notebook.py                # Automated Python script to generate/regenerate the complete notebook
│
├── dataset/                         # PlantVillage dataset structured by category (50 images/class starter subset)
│   ├── Potato___Early_blight/       # Alternaria solani on potato foliage
│   ├── Potato___Late_blight/        # Phytophthora infestans on potato foliage
│   ├── Potato___healthy/            # Disease-free healthy potato leaves
│   ├── Tomato___Early_blight/       # Alternaria solani on tomato foliage
│   ├── Tomato___Late_blight/        # Phytophthora infestans on tomato foliage
│   └── Tomato___healthy/            # Disease-free healthy tomato leaves
│
└── saved_models/                    # Exported model weights and metadata
    ├── plant_disease_mobilenetv2.keras   # Full trained Keras model (architecture + weights + optimizer)
    ├── best_plant_model.weights.h5       # Best checkpoint weights during training
    └── class_indices.json                # JSON mapping of class indices to disease names
```

---

## 🎯 Target Categories & Pathogens

| # | Class Name | Crop | Health Status | Pathogen / Agent | Typical Symptoms |
|---|---|---|---|---|---|
| 0 | `Potato___Early_blight` | Potato | Diseased | *Alternaria solani* (Fungus) | Concentric dark brown target spots, chlorotic yellow halos |
| 1 | `Potato___Late_blight` | Potato | Diseased | *Phytophthora infestans* (Oomycete) | Water-soaked greasy lesions, rapid wilting, white mildew |
| 2 | `Potato___healthy` | Potato | Healthy | None (Optimal) | Uniform green foliage, intact cellular lamina |
| 3 | `Tomato___Early_blight` | Tomato | Diseased | *Alternaria solani* (Fungus) | Brown ring spots starting on lower leaves, defoliation |
| 4 | `Tomato___Late_blight` | Tomato | Diseased | *Phytophthora infestans* (Oomycete) | Dark brown necrosis on leaves, stems, and fruits |
| 5 | `Tomato___healthy` | Tomato | Healthy | None (Optimal) | Vibrant green leaves, strong vegetative growth |

---

## 🧠 Deep Learning Architecture

- **Backbone**: MobileNetV2 (Pre-trained on ImageNet, 1.4M images)
- **Input Dimension**: $224 \times 224 \times 3$
- **Preprocessing**: Integrated Rescaling layer ($[0, 255] \to [-1, 1]$)
- **Data Augmentation**: `RandomFlip`, `RandomRotation(0.15)`, `RandomZoom(0.15)`, `RandomContrast(0.10)`
- **Classification Head**:
  - `GlobalAveragePooling2D`
  - `BatchNormalization`
  - `Dense(256, activation='relu')`
  - `Dropout(0.3)`
  - `Dense(6, activation='softmax')`
- **Training Strategy**:
  - **Phase 1 (Feature Extraction)**: Base weights frozen; top classifier head trained with Adam ($\text{lr} = 10^{-3}$) for 8 epochs.
  - **Phase 2 (Fine-Tuning)**: Unfreezing top backbone layers from layer 100 onwards with micro learning rate ($\text{lr} = 10^{-5}$) for 4 epochs.

---

## 📊 Performance Metrics (Test Set)

| Metric | Score |
|---|---|
| **Overall Test Accuracy** | **93.33%** |
| **Test Categorical Loss** | **0.4286** |
| **Macro Average F1-Score** | **0.9333** |
| **Weighted Average F1-Score** | **0.9337** |

### Detailed Classification Breakdown:
```text
                       precision    recall  f1-score   support
Potato___Early_blight      1.000     0.909     0.952        11
Potato___Late_blight       0.800     1.000     0.889         8
Potato___healthy           1.000     1.000     1.000         6
Tomato___Early_blight      0.889     0.889     0.889         9
Tomato___Late_blight       0.909     0.833     0.870        12
Tomato___healthy           1.000     1.000     1.000        14

accuracy                                       0.933        60
macro avg                  0.933     0.939     0.933        60
weighted avg               0.938     0.933     0.934        60
```

---

## 🚀 How to Run the Project

### 1. Launch Jupyter Notebook
Open PowerShell or your command prompt in this directory (`d:\Plant Disease Detection`):

```bash
python -m notebook Plant_Disease_Detection.ipynb
```
*Alternatively, you can open the folder directly in VS Code and open `Plant_Disease_Detection.ipynb`.*

### 2. Sequential Cell Execution
All 20 sections are organized sequentially. You can click **Run All** or step through cells with `Shift + Enter`:
1. **Section 1**: Environment setup, imports, and GPU diagnostics.
2. **Section 2**: Hyperparameters & configurations.
3. **Section 3**: Dataset verification and automated downloader.
4. **Section 4**: Exploratory data analysis (EDA) & Seaborn distribution chart.
5. **Section 5**: Sample leaf visualization grid across all 6 classes.
6. **Section 6**: Preprocessing pipeline and 70/15/15 train/val/test split.
7. **Section 7**: Data augmentation and visual comparison.
8. **Section 8**: MobileNetV2 transfer learning architecture construction.
9. **Section 9**: Model compilation and callbacks (EarlyStopping, ReduceLROnPlateau).
10. **Section 10**: Phase 1 training (Feature Extraction).
11. **Section 11**: Phase 2 training (Fine-Tuning).
12. **Section 12**: Training accuracy and loss dynamics plots.
13. **Section 13**: Final test evaluation on unseen leaf samples.
14. **Section 14**: Scikit-Learn classification report (Precision, Recall, F1-Score).
15. **Section 15**: Annotated Seaborn confusion matrix heatmap.
16. **Section 16**: Model serialization to `.keras` and `class_indices.json`.
17. **Section 17**: Single-leaf inference pipeline with agronomic remedy recommendations.
18. **Section 18**: Batch testing gallery with color-coded Correct/Incorrect titles.
19. **Section 19**: Standalone model reload and verification test.
20. **Section 20**: Edge deployment roadmap & TFLite conversion guide.

---

## 🌾 Agronomic Disease Diagnostic & Remedy Example

When you run `predict_and_diagnose_leaf(image_path, model, class_names)`, the notebook displays:
1. **The input leaf image** alongside a **Top-3 confidence bar chart**.
2. **Pathogen Identification**: Biological classification (Fungus vs Oomycete vs Healthy).
3. **Cultural Control**: Crop rotation, pruning, drip irrigation, sanitization.
4. **Chemical / Organic Treatments**: Targeted recommendations (e.g. Copper fungicides, Chlorothalonil, Mancozeb, Neem bio-stimulants).
