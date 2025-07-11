from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import PredictPipeline, Customdata
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

## Route to the home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        data = Customdata(
            gender=request.form.get('gender'),
            race_enthicity=request.form.get('race_enthicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course')

        )

        pred_df = data.get_data_as_dataframe()

        print(pred_df)

        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0])
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)



