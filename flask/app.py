import pandas as pd
from keras.models import load_model
import h5py
#prd_model = load_model('../data/model_weights.h5')
import os

from flask import Flask, url_for, request, render_template, session, jsonify,redirect
app = Flask(__name__)
app.config['IMAGE_UPLOADS'] = '/Volumes/T5_500G/Capstone/v2/flask/static/uploads'
app.config['SECRET_KEY'] = 'mykey'

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():

    prediction_results = {}
    animals = ['% Dog Score', '% Cat Score']
    animals_scores = []
    imageUrl = ""

    if request.method == 'POST':
        
        if request.files:
            import keras
            from keras.models import load_model
            from keras import backend as K
            import numpy as np
            model = load_model('/Volumes/T5_500G/Capstone/v2/flask/models/model_weights.h5')
            print(request.files)
            print(request.files['image'].filename)
            imageUrl = "/static/uploads/"+request.files['image'].filename
            image2 = app.config['IMAGE_UPLOADS']+"/"+request.files['image'].filename
            request.files['image'].save(image2)
            #image = "/Volumes/T5_500G/Capstone/v2/flask/uploads/dog.png"
            img_path = image2
            img = keras.preprocessing.image.load_img(img_path, target_size=(224,224))
            img_array = keras.preprocessing.image.img_to_array(img)
            expanded_img_array = np.expand_dims(img_array, axis=0)
            preprocessed_img = expanded_img_array / 255. # Preprocess the image
            prediction = model.predict(preprocessed_img)
            pred_list = prediction.tolist()
            print(pred_list)
            # prediction_results['cat_score'] = pred_list [0][0]
            # prediction_results['dog_score'] = pred_list [0][1]
            animals_scores.append(pred_list[0][0])
            animals_scores.append(pred_list[0][1])
            #animals_scores[1] = pred_list [0][0]
            # prediction_json = jsonify(prediction_results)
            print(prediction_results)
            #print(prediction_json)

            imageUrl = "/static/uploads/"+request.files['image'].filename
    return render_template('/upload_image.html',imageUrl=imageUrl, animals=animals, animals_scores=animals_scores)



@app.route('/line_chart')
def line_chart():
    return render_template('line_chart.html')




if __name__ == '__main__':
    app.run(debug=True)