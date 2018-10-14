import os
import sys
import logging
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS

from model import Classifier, get_data


app = Flask(__name__)
CORS(app)

model_SA = Classifier('model/sa.pb', 'SA')
model_BKNET = Classifier('model/bknet.pb', 'BKNET')

@app.route('/api/predict', methods=['POST'])
def predict():
    content = request.get_json()
    # {"input":"1,2,3,4,5","model":"SA"}
    x = get_data(content["input"])
    output = None
    if content["model"] == "SA":
        output = model_SA.predict(x)
    elif content["model"] == "BKNET":
        output = model_BKNET.predict(x)
    print('>>>>>>> predict done!!!')
    if output is not None:
        output = ",".join(output.astype(str))
        print('>>>>>>>>>>> ouput is not none', output)
        return jsonify({"message": "OK", "data": output})
    else:
        print('>>>>>>>>>>>>>>>>>>>output is none')
        return jsonify({"message": "Wrong argument"})

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    # This is used when running locally.
    app.run(host='0.0.0.0', debug=True)