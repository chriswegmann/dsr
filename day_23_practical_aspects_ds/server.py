from flask import Flask, jsonify, request
import pandas as pd
from sklearn.externals import joblib

app = Flask(__name__)

loaded_model = joblib.load('model.pkl')
imputed_values = pd.read_json('imputed_values.json', typ='series')
imputed_values = pd.DataFrame(imputed_values).transpose()

@app.route('/predict', methods=['POST'])
def predict():

    if request.json['basket']:
        basket = request.json['basket']
    else:
        basket = imputed_values.loc[0,'basket']

    if request.json['zipCode']:
        zipCode = request.json['zipCode']
    else:
        zipCode = imputed_values.loc[0,'zipCode']

    if request.json['totalAmount']:
        totalAmount = request.json['totalAmount']
    else:
        totalAmount = imputed_values.loc[0,'totalAmount']

    p = probability(basket, zipCode, totalAmount)
    return jsonify({'probability': p}), 201

def probability(basket, zipCode, totalAmount):

    print("Processing request: {},{},{}".format(basket, zipCode, totalAmount))

    df = pd.DataFrame(columns=['basket', 'zipCode', 'totalAmount', 0, 1, 2, 3, 4])
    df = df.append({'basket': basket, 
                'zipCode': zipCode, 
                'totalAmount': totalAmount, 
                0:0, 1:0, 2:0, 3:0, 4:0}, 
                ignore_index=True)

    for basket in set.union(*df.basket.apply(set)):
        df[basket] = df.apply(lambda _: int(_.basket.count(basket)), axis=1)
    df.drop('basket', axis=1, inplace=True)    

    df["zipCode"] = df["zipCode"].astype('category',categories=[i for i in list(range(1000,10000))])
    dummies = pd.get_dummies(df.zipCode)
    df = pd.concat([df, dummies], axis=1)
    df.drop('zipCode', axis=1, inplace=True)

    predictions = loaded_model.predict_proba(df)

    return float(predictions[:,1])