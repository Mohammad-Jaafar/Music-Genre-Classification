# Music Genre Classification (GTZAN Dataset)

This project focuses on classifying music genres using GTZAN dataset.  
It demonstrates two major approaches in audio classification:

- **Tabular Learning** using MFCC features + Random Forest  
- **Image-based Deep Learning** using Spectrograms + CNN / VGG16 Transfer Learning  

The project highlights how audio signals can be converted into meaningful numerical and visual features, then used to train robust multi-class classification models.

---

## Overview

The goals of this project are:

- Load and explore the **GTZAN music genre dataset**
- Extract meaningful audio features using **Librosa**:
  - MFCCs (Mel-frequency cepstral coefficients)
  - Mel-spectrogram images
- Train and evaluate multiple machine learning models:
  - **Random Forest** (tabular MFCC-based features)
  - **CNN built from scratch** (spectrogram images)
  - **VGG16 Transfer Learning** (spectrogram images)
- Compare model performance across different learning paradigms
- Save trained deep learning models for later inference

---

## Features

-   Extract MFCC features and train a **Random Forest** classifier.
-   Generate **Mel-Spectrogram images** for CNN and VGG16 models.
-   Train and evaluate **CNN from scratch** and **VGG16 Transfer Learning**.
-   Compare model performance using **accuracy**, **validation loss**, and **classification reports**.
-   Visualize training progress and model predictions.

---

## Technologies Used

-   **Python 3.9+**
-   **Librosa** (audio feature extraction)
-   **NumPy / Pandas**
-   **Matplotlib / Seaborn**
-   **TensorFlow / Keras**
-   **Scikit-learn**

---

## Project Structure
```
MusicGenreClassification/
├── genre_classification.ipynb      # Jupyter Notebook
├── README.md                        # Project documentation
└── requirements.txt                 # Python dependencies
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Mohammad-Jaafar/Music-Genre-Classification.git
   ```
2. Place the GTZAN dataset in:
   ```
   Data/genres_original/
   ```
3. Open the notebook in Jupyter or Google Colab.
4. Run all cells step-by-step to reproduce the results.
---

## Results & Training Performance

### CNN (Spectrogram-based)
- Final validation accuracy: **48%**
- Accuracy improvement across epochs:
  - Starts around 13%
  - Ends around 48%
- Best performance achieved around epoch 20

### VGG16 Transfer Learning
- Final validation accuracy: **55%**
- Higher accuracy than CNN from scratch
- Learns faster and generalizes better

### Random Forest (MFCC-based)
- Accuracy varies depending on MFCC quality  
- Provides a strong baseline for comparison

**Conclusion:**  
Transfer learning using **VGG16** achieves the best results on spectrogram images.

---

## Model Comparison

| Model | Input Type | Accuracy | Notes |
|------|------------|----------|-------|
| Random Forest | MFCC (tabular) | Moderate | Good baseline, fast to train |
| CNN (scratch) | Spectrogram images | 48% | Learns local time-frequency patterns |
| VGG16 Transfer Learning | Spectrogram images | 55% | Best model, benefits from pre-trained features |


---

## Author
**Mohammad Jaafar**  
mhdjaafar24@gmail.com  
[LinkedIn](https://www.linkedin.com/in/mohammad-jaafar-)  
[HuggingFace](https://huggingface.co/Mhdjaafar)  
[GitHub](https://github.com/Mohammad-Jaafar)

---

*If you find this project helpful, please consider starring the repository on GitHub!*  
