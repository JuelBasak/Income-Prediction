from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1000 * 1000

@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

@app.route('/predict_data', methods=['POST'])
def predict_data():
    age = request.form['age']
    workclass = request.form['workclass']
    fnlwgt = request.form['fnlwgt']
    education = request.form['education']
    education_num = request.form['education_num']
    marital_status = request.form['marital_status']
    occupation = request.form['occupation']
    relationship = request.form['relationship']
    race = request.form['race']
    sex = request.form['sex']
    capital_gain = request.form['capital_gain']
    capital_loss = request.form['capital_loss']
    hours_per_week = request.form['hours_per_week']
    native_country = request.form['native_country']

    # print(age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country)
    data = pd.DataFrame(data=[[age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country]],
    columns=['age', 'workclass', 'fnlwgt', 'education', 'education_num',
       'marital_status', 'occupation', 'relationship', 'race', 'sex',
       'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'])

    # print(data)

    output = preprocess_and_model(data)

    data = data.reset_index()

    return render_template('output.html', data=data.values, columns=data.columns, output=output)

@app.route('/predict_file', methods=['POST'])
def predict_file():
    file = request.files['file']

    if file.filename.rsplit('.')[-1] not in ['csv', 'CSV']:
        return render_template('index.html', output='Invalid file. Allowed files [CSV]')

    data = pd.read_csv(file)
    output = preprocess_and_model(data)

    data = data.reset_index()
    
    return render_template('output.html', data=data.values, columns=data.columns, output=output)


def preprocess_and_model(df):
    full_pipeline = joblib.load('pipelines/02-10-2022_19-41-10_full_pipeline.pkl')
    xgb_clf = joblib.load('model/02-10-2022_19-38-27_xgb_clf.pkl')

    prepared_data = full_pipeline.transform(df)
    prediction = xgb_clf.predict(prepared_data)

    target_encoder = joblib.load('pipelines/02-10-2022_19-41-10_target_encoder.pkl')

    target_value = target_encoder.inverse_transform(prediction)

    # print(target_value)

    return target_value

if __name__ == '__main__':
    app.run(debug=False)

    