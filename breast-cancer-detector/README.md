# Breast Cancer Diagnosis App

## About

This ML-powered Breast Cancer Diagnosis App is assists medical professionals in diagnosing breast cancer. With a set of cytology measurements, the app can classify a breast mass as "Benign" or "Malignant". A radar visualization of the manual input data, the predicted diagnosis and probabilities by type is shown in th UI. Connection to a laboratory cytology machine can also be used for predictions, but it is out of scope for this project.

## Data

This is a machine learning exercice from the public dataset [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). 

**Note:** This dataset is for educational purposes and not for professional use.

## Demo App

A live version of the application can be found on [Streamlit Community Cloud](link).

## Installation

You can run this inside a virtual environment to make it easier to manage dependencies. Recommend using `conda` to create a new environment and install the required packages. You can create a new environment called `breast-cancer-diagnosis` by running:

```bash
conda create -n breast-cancer-diagnosis python=3.10 
```

Then, activate the environment:

```bash
conda activate breast-cancer-diagnosis
```

Then, activate the environment:

```bash
conda activate breast-cancer-diagnosis
```

To install the required packages, run:

```bash
pip install -r requirements.txt
```

This will install all the necessary dependencies, including Streamlit, OpenCV, and scikit-image.

## Usage
To start the app, simply run the following command:

```bash
streamlit run app/main.py
```

You can then input the various cell measurments to see the prediction in action.

##### Credits
https://github.com/alejandro-ao