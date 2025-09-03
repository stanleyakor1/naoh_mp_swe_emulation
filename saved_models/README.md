
Saved Models Update
This commit adds several new trained PyTorch models to the saved_models directory. These models were saved from various training routines, each using a different combination of input features to explore their impact on the final model performance.

Models Added
The following models have been added to the repository:

WRF_MODIS_STATIC_v1.pth

WRF_MODIS_STATIC_v2.pth

WRF_MODIS_v1.pth

WRF_v1.pth

Model Naming Convention
The name of each model file is designed to clearly describe the set of input features used for its training. The naming convention is as follows:

Name Component

Description

WRF

Trained with WRF (Weather Research and Forecasting) atmospheric forcings.

MODIS

Trained with MODIS (Moderate Resolution Imaging Spectroradiometer) data, specifically snow cover and Leaf Area Index (LAI).

STATIC

Trained with static variables, including elevation and temporal encoding.

vX

A version number indicating different training runs (e.g., v1, v2).

Model Descriptions
WRF_MODIS_STATIC: This model was trained on a comprehensive set of features, including WRF atmospheric forcings, MODIS snow cover and LAI, and static variables like elevation and temporal encoding. The v1 and v2 versions represent distinct training runs.

WRF_MODIS: This model was trained using WRF atmospheric forcings and MODIS snow cover.

WRF: This model was trained exclusively on WRF atmospheric forcings.
