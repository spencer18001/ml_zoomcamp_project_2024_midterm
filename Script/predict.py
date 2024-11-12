import pickle
import xgboost as xgb
from flask import Flask
from flask import request
from flask import jsonify

target_name = "Win rate"
model_file = "model.bin"

with open(model_file, "rb") as f_in:
    model_name, dv, model = pickle.load(f_in)

features = list(dv.get_feature_names_out())

app = Flask("lol")

@app.route('/predict', methods=['POST'])
def predict():
    player = request.get_json()

    X = dv.transform([player])

    if model_name == "linear regression":
        y_pred = model.predict(X)
    elif model_name == "decision tree":
        y_pred = model.predict(X)
    elif model_name == "random forest":
        y_pred = model.predict(X)
    elif model_name == "xgboost":
        dmatrix = xgb.DMatrix(X, feature_names=features)

        y_pred = model.predict(dmatrix)
        
    result = {
        target_name: y_pred[0],
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
