# Music Genre Classification (GTZAN Dataset)

This project focuses on classifying music genres using GTZAN dataset.  
It demonstrates two major approaches in audio classification:

- **Tabular Learning** using MFCC features + Random Forest  
- **Image-based Deep Learning** using Spectrograms + CNN / VGG16 Transfer Learning  

The project highlights how audio signals can be converted into meaningful numerical and visual features, then used to train robust multi-class classification models.

---

## Project Overview

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

## Machine Learning Techniques Used

### Tabular Audio Classification
- **MFCC Feature Extraction**  
- **Random Forest Classifier**

### Image-based Audio Classification
- **Mel-Spectrogram Generation**
- **Convolutional Neural Network (CNN)**
- **VGG16 Transfer Learning Model**

### Evaluation Metrics
- Accuracy  
- Validation Loss  
- Epoch-based Training Curves  
- Classification Report (for MFCC-based model)

---

## Dataset

- **Name:** GTZAN – Music Genre Classification Dataset  
- **Classes:** 10 genres  
  (e.g., classical, pop, rock, jazz, metal, disco, reggae, hiphop, blues, country)  
- **Files:** 1000 audio tracks (100 per class)  
- **Format:** WAV  
- **Duration:** ~30 seconds each  

The dataset must be placed under:
```
Data/genres_original/
```

Each folder should contain `.wav` files for one genre.

---

## Project Structure
```
MusicGenreClassification/
│
├── genre_classification.ipynb      # Jupyter Notebook
├── README.md                        # Project documentation
└── requirements.txt                 # Python dependencies
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Mohammad-Jaafar/Music-Genre-Classification.git
cd Music-Genre-Classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

All necessary libraries such as **Librosa**, **Scikit-Learn**, **TensorFlow**, and **Matplotlib** will be installed.

### 3. Run the Notebook / Script
To open the notebook:
```bash
jupyter notebook genre_classification.ipynb
```

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

## Future Improvements

- Hyperparameter tuning for CNN and Random Forest  
- Data augmentation for spectrograms  
- Training deeper architectures (ResNet, EfficientNet)  
- Deploying as a web app using **Streamlit** or **Gradio**  
- Adding prediction script for user-uploaded audio files  

---

## Author

**Mohammad Jaafar**  
mhdjaafar24@gmail.com  
[LinkedIn](https://www.linkedin.com/in/mohammad-jaafar-)  
[GitHub](https://github.com/Mohammad-Jaafar)

---

*If you find this project helpful, please consider starring the repository on GitHub!*  
