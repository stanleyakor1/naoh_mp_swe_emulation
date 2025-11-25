# Noah-MP SWE Emulation 
ConvLSTM emulator of Noah-MP SWE with preprocessing, training, and evaluation tools, supporting remote-sensing inputs.

## Overview  
This repository implements a deep learning (ConvLSTM) framework to emulate the snow water equivalent (SWE) output of the Noah‑MP land surface model in mountainous terrain. The primary target region is the Colorado Rockies, but the code is structured generically to support similar high-latitude or mountainous domains.  
Key features:  
- Preprocessing of model forcing, terrain and remote-sensing predictors  
- ConvLSTM2D  model architecture for spatiotemporal SWE emulation  
- Model training pipeline, inference, and performance evaluation  
- Support for remote-sensing inputs (e.g., snow cover area, leaf area index) and model output comparands  
- Saved model management and reproducibility tools  

## Motivation & Scope  
Simulating SWE in mountainous regions is computationally intensive when using full physics-based land surface models. By training a deep learning emulator, we can:  
- Exploit abundant model output from multiple forcing experiments (e.g., WRF runs)  
- Forecast SWE at high spatial resolution with reduced computational cost  
- Explore spatiotemporal dependencies via ConvLSTM architectures  
- Provide a tool for operational or research forecasting of snow accumulation and melt in mountain basins  

## Repository Structure
```
├── notebooks/                 # Jupyter notebooks for exploratory analysis
├── saved_models/              # Pre-trained model weights and run logs
├── snotel/                    # SNOTEL data download and preprocessing scripts
│   └── DOWNLOAD_SNOTEL
├── src/                       # Core Python modules
│   ├── data_preprocessing.py
│   ├── model_architecture.py
│   ├── training.py
│   ├── evaluation.py
│   └── utilities.py
├── run_train.py               # Entry point for training the SWE emulator
├── requirements.txt           # Python dependencies
├── setup.py                   # Optional package installer
└── LICENSE                    # Project license
```  

## Installation  
1. Clone this repository:  
   ```bash
   git clone https://github.com/stanleyakor1/naoh_mp_swe_emulation.git
   cd naoh_mp_swe_emulation
