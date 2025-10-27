# 🌾 DNA Base Crop Selector
**AI-Powered Genomic Prediction for Major Crops using the Cropformer Deep Learning Framework**

---

## 🧬 Overview

**DNA Base Crop Selector** is a **Flask-based web application** that provides an intuitive interface for predicting crop performance using genomic data.

By leveraging the **Cropformer** deep learning architecture, this tool enables researchers, breeders, and students to assess the genetic potential of crop varieties with ease and precision.

The system implements concepts from the paper:  
> *“Cropformer: An interpretable deep learning framework for crop genomic prediction”*

The application supports predictions for **five major crops**, each powered by a specialized model:

| Crop | Predicted Trait | Model Type |
|------|------------------|-------------|
| 🌽 **Maize** | Days to Tasseling (DTT) | Genetics-Only |
| 🌾 **Wheat** | Thousand Kernel Weight (TKW) | Gene × Environment (GxE) |
| 🍅 **Tomato** | Fruit Weight (FW) | Genetics-Only |
| 🌿 **Foxtail Millet** | Grain Yield | Gene × Environment (GxE) |
| 🌾 **Rice** | Grain Yield | Genetics-Only |

---

## ⚙️ Core Technology

The predictive power of this application is based on the **Cropformer hybrid deep learning architecture**, which integrates:

### 🧩 1. Convolutional Neural Networks (CNNs)
Extract meaningful features and local patterns automatically from the raw SNP (genetic marker) data.

### 🧠 2. Multi-Head Self-Attention (Transformers)
Captures complex, long-range genetic interactions and learns which markers influence the target trait most strongly.

### Model Categories
- **Genetics-Only Models:** Use only genetic (SNP) data.  
  *(Maize, Tomato, Rice)*
- **Gene × Environment (GxE) Models:** Combine genetic data with environmental features for enhanced prediction.  
  *(Wheat, Foxtail Millet)*

---

## 🌟 Key Features

- 🧑‍💻 **User-Friendly Web Interface** – Clean, minimal UI for uploading data and viewing results.
- 🌾 **Multi-Crop Support** – Separate, fine-tuned prediction pipelines for five different crops.
- 🔄 **Automatic Model Downloading** – Automatically fetches `.pth` (model) and `.pkl` (scaler) files from Google Drive.
- 📊 **Insight Generation** – Highlights the “standout sample” based on the desired agronomic trait  
  *(lowest DTT for Maize, highest yield for others)*.
- 📥 **Downloadable Results** – Export predictions as a CSV file for further analysis.

---

## 📁 Project Structure

```bash
cropformer-webapp/
│
├── app.py # Main Flask app (routing + web logic)
│
├── foxtail_millet_predictor.py # GxE model logic + downloader
├── rice_predictor.py # Genetics-only model logic + downloader
├── tomato_predictor.py # Genetics-only model logic + downloader
│
├── generate_samples.py # Generates synthetic data for testing
├── create_tomato_preprocessors.py # Creates scaler & PCA files for tomato model
│
├── models/ # Folder for trained model files (.pth)
│ └── (Downloaded automatically)
│
├── scalers/ # Folder for preprocessor files (.pkl)
│ └── (Downloaded automatically)
│
├── templates/
│ └── index.html # Single-page HTML web interface
│
├── uploads/ # Temporary folder for user-uploaded CSVs
│
└── requirements.txt # Required Python packages
```


## 🧰 Setup and Installation

### Prerequisites
- Python **3.8+**
- `pip` (Python package installer)

---

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd cropformer-webapp
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. (Optional) Update Google Drive Links
The predictors for Foxtail Millet and Rice automatically download model files.
If you have your own models hosted on Google Drive, update their IDs:

In foxtail_millet_predictor.py, modify the FILES_TO_DOWNLOAD dictionary.

In rice_predictor.py, modify the FILES_TO_DOWNLOAD dictionary.

### 4. Place Core Model Files
Ensure the models and scalers are in the correct directories:

```bash
models/maize_model.pth
models/wheat_model.pth

scalers/maize_scaler.pkl
scalers/wheat_geno_scaler.pkl
scalers/wheat_env_scaler.pkl
```

### 🚀 How to Run
Once setup is complete, start the Flask web server:

```bash
python app.py
```
Then open your browser and visit:

```cpp
http://127.0.0.1:5000/
```
You can upload your genomic CSV file, choose the crop, and view the predicted results instantly.

### 🧠 Citation
If you use this application or concept in your work, please cite the original paper:

Wang, H., Yan, S., Wang, W., et al. (2025). Cropformer: An interpretable deep learning framework for crop genomic prediction. Plant Communications, 6, 101223. 


DOI: https://doi.org/10.1016/j.xplc.2024.101223
