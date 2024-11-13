# LoL Championship Player Performance Prediction

This project analyzes player performance data from the **2024 League of Legends World Championship** to predict win rates based on in-game statistics. By modeling the 'Win rate' as the target variable, the project aims to uncover which gameplay strategies correlate with higher success rates.

## Dataset
[**2024 League of Legends World Championship**](https://www.kaggle.com/datasets/anmatngu/2024-lol-championship-player-stats-and-swiss-stage)

The dataset includes detailed player statistics, such as:
- **KDA (Kill/Death/Assist) Ratio**
- **Vision Score per Minute (VSPM)**
- **Wards Placed per Minute (WPM)**
- **Gold Differential at 15 Minutes (GD@15)**
- ... etc.

Each row represents a player’s cumulative performance metrics throughout the championship, offering insights into individual impact and strategic effectiveness.

## `Notebook/notebook.ipynb`

### 1. Exploratory Data Analysis (EDA)
- **Data Cleaning**: Removed unnecessary columns and refined specific numerical features.
- **Feature Analysis**: Visualized numerical and categorical feature distributions using histograms, bar plots, and correlation heatmaps.
- **Feature Importance**: Assessed feature relevance with correlation analysis and `mutual_info_regression`.
- **Data Preparation**: Split dataset using k-fold cross-validation and encode categorical features.

### 2. Model Training and Tuning
Trained and tuned several models with parameter optimization:
- **Linear Regression**: parameter `alpha`.
- **Decision Tree**: parameters `max_depth` and `min_samples_leaf`.
- **Random Forest**: parameters `n_estimators`, `max_depth`, and `min_samples_leaf`.
- **XGBoost**: parameters `eta`, `max_depth`, and `min_child_weight`.

### 3. Model Selection
Evaluated models using RMSE as the metric. The best model’s name and parameter settings are stored in `best_model_params.json`.
- The RMSE of the test dataset:
```
0.1450606188503998
```
## `Pipenv`
This project uses **Pipenv** for managing dependencies in the development environment.


```bash
pip install pipenv # install

pipenv --version # verify the installation
# my output: pipenv, version 2024.4.0

pipenv install # install project dependencies

pipenv shell # activate the Pipenv virtual environment
```

## `Script/train.py`
Train and save the model:
- Preprocesses data and saves the test dataset to `test_players.json`.
- Loads the best model's name and parameters from `best_model_params.json`.
- Trains the model and saves the model name, encoder, and model to `model.bin`.

## `Script/predict.py`
This script provides a prediction service for player win rates using a trained model.
- Sets up a prediction service using Flask.
- Loads the trained model from `model.bin`.
- Receives player data through requests and returns win rate predictions in JSON format.

```bash
python Script/train.py
```

## `Dockerfile`
This Dockerfile sets up a containerized environment for the win rate prediction service.
- Builds a container that runs the prediction service using Flask.
- Exposes the service on port 9696 for API requests.

```bash
docker build -t lol-prediction .
docker run -it -p 9696:9696 lol-prediction
```

## `Script/predict_test.py`
This script tests the prediction service by sending requests using player data from the test dataset.
- Reads player data from `test_players.json`.
- Selects a random player, sends a request to the prediction service, and prints the response.

```bash
python Script/predict_test.py
```