import json, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb

df_path = "Data/player_statistics_cleaned_final.csv"
target_name = "Win rate"
seed = 123
test_data_file = "test_players.json"
best_model_file = "best_model_params.json"
model_file = "model.bin"

### Data Preprocessing
df = pd.read_csv(df_path)

# Drop columns with high correlation to the target ('Win rate') to avoid relying on non-generalizable features
df.drop(columns=["TeamName", "PlayerName", "Country"], inplace=True)

# Convert "Solo Kills" to a numerical feature:
# According to the dataset documentation, "-" likely represents 0
df["Solo Kills"] = df["Solo Kills"].replace(to_replace="-", value=0)
df["Solo Kills"] = df["Solo Kills"].astype(int)

# Drop the "Penta Kills" column since it contains only zeros and adds no predictive value
df.drop(columns=["Penta Kills"], inplace=True)

# Using only df_full_train and df_test due to small dataset size and planned cross-validation
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train[target_name].values
y_test = df_test[target_name].values

del df_full_train[target_name]
del df_test[target_name]

# Encode features using DictVectorizer to convert data into numerical format for model training
dv = DictVectorizer(sparse=False)

full_train_dict = df_full_train.to_dict(orient="records")
X_full_train = dv.fit_transform(full_train_dict)

test_dict = df_test.to_dict(orient="records")
X_test = dv.fit_transform(test_dict)

features = list(dv.get_feature_names_out())

# Save test dataset for prediction service testing
json_object = json.dumps(test_dict, indent=2)
json_object = json_object.replace("\n    ", "").replace("\n  }", "}")
with open(test_data_file, "w") as outfile:
    outfile.write(json_object)

### Training
# Load the best model's name and hyperparameters
with open(best_model_file, "r") as infile:
    json_object = json.load(infile)

best_model_name = json_object["model_name"]
param_dict = json_object["params"]

print("training the final model")

# Select the best model based on the name and hyperparameters, then train and make predictions on the test set
if best_model_name == "linear regression":
    if param_dict["alpha"] == 0:
        model = LinearRegression()
    else:
        model = Ridge(**param_dict, random_state=seed)
    model.fit(X_full_train, y_full_train)
    y_pred = model.predict(X_test)
elif best_model_name == "decision tree":
    model = DecisionTreeRegressor(**param_dict, random_state=seed)
    model.fit(X_full_train, y_full_train)
    y_pred = model.predict(X_test)
elif best_model_name == "random forest":
    model = RandomForestRegressor(**param_dict, random_state=seed, n_jobs=-1)
    model.fit(X_full_train, y_full_train)
    y_pred = model.predict(X_test)
elif best_model_name == "xgboost":
    dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train,
                    feature_names=features)
    dtest = xgb.DMatrix(X_test, feature_names=features)

    params = {
        "seed": seed,
        "objective": 'reg:squarederror',
        "nthread": 8,
        "verbosity": 1,
    } | param_dict
    model = xgb.train(params, dfulltrain, num_boost_round=200)
    y_pred = model.predict(dtest).tolist()

# Calculate RMSE on test dataset
print(f"The RMSE of the test dataset: {root_mean_squared_error(y_test, y_pred)}")

# Save the best model name, DictVectorizer, and model to a file using pickle
with open(model_file, "wb") as f_out:
    pickle.dump((best_model_name, dv, model), f_out)

print(f"the model is saved to {model_file}")
