import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle, json


app = Flask(__name__)
with open('fraud_detection_saved_model.pkl','rb') as f:
    model = pickle.load(f)

feature_keys = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
"V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18","V19","V20",
"V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods = ['POST'])
def predict():

    data = json.loads(list(request.form.values())[0])
    features = []
    for key in feature_keys:
        if key in data:
            features.append(float(data[key]))
        else:
            features.append(0)


    # features = [float(data[key]) for key in feature_keys]
    features_np = np.array(features).reshape(1, -1)
    prediction = model.predict(features_np)
    print(prediction[0])
    class_label = "Normal" if prediction[0] == 0 else "Fraud"
    if prediction[0] == 0:
        message = "[*] The transaction is Normal."
    else:
        message = "[!] The transaction seems Fraud."
    request.get_json()
    return render_template('home.html', prediction_text=f"{message}")

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

# def load_features(data):
#     output = []
#     for key in feature_keys:
#         output.append(float(data.get(key, 0))
#     return output
        



if __name__ == '__main__':
    app.run(debug=True)