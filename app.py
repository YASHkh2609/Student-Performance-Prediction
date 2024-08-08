from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
application = Flask(__name__)

app = application

#route for a home page
@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('home.html')#will have the fields for the inputs to be provided from the user
    else:
        data = CustomData(gender=request.form.get('gender'),
                          race_ethnicity=request.form.get('race_ethnicity'),
                          parental_level_of_education=request.form.get('parental_level_of_education'),
                          lunch=request.form.get('lunch'),
                          test_preparation_course=request.form.get('test_preparation_course'),
                          reading_score=float(request.form.get('reading_score')),
                          writing_score=float(request.form.get('writing_score'))
                          )
        #calling method to convert inputs into dataframe
        custom_df = data.get_data_as_data_frame() 
        print(custom_df)
        #give df to the predict method to perform transformation and prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(custom_df)

        return render_template('home.html', results=results[0])
    
if __name__=="__main__":
    # app.run(port= 5000, debug=True)
    app.run(port= 5000)# before deployment we should remove the debug mode

