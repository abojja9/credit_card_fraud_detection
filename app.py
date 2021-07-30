import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle, json


app = Flask(__name__)
with open('model_dir/fraud_detection_saved_model.pkl','rb') as f:
    model = pickle.load(f)

feature_keys = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
"V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18","V19","V20",
"V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods = ['POST'])
def predict():
    try:
        data = json.loads(list(request.form.values())[0])
        features = []
        for key in feature_keys:
            if key in data:
                features.append(float(data[key]))
            else:
                features.append(0)

        features_np = np.array(features).reshape(1, -1)
        prediction = model.predict_proba(features_np)[0]
        class_label = "Normal" if prediction[1] < 0.90 else  "Fraud" 
        
        if class_label == "Normal":
            message = "[*] The transaction is Normal."
        else:
            message = "[!] The transaction seems Fraud."
    except:
            message = "The model failed"
    return render_template('home.html', prediction_text=f"{message}")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    try:
        data = request.get_json(force=True)
        features = []
        for key in feature_keys:
            if key in data:
                try:
                    features.append(float(data[key]))
                except:
                    features.append(0)
            else:
                features.append(0)

        features_np = np.array(features).reshape(1, -1)
        prediction = model.predict_proba(features_np)[0]
        class_label = "Normal" if prediction[1] < 0.90 else  "Fraud" 

        if class_label == "Normal":
            message = "[*] The transaction is Normal."
        else:
            message = "[!] The transaction seems Fraud."
    
    except:
        message = "The model failed"
    return json.dumps({"message": message})


if __name__ == '__main__':
    app.run(debug=True)