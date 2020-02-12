import pandas as pd
from keras.models import load_model
import h5py
#prd_model = load_model('../data/model_weights.h5')
import os

from flask import Flask, url_for, request, render_template, session, jsonify,redirect
app = Flask(__name__)
app.config['IMAGE_UPLOADS'] = '/Volumes/T5_500G/Capstone/v2/flask/uploads'

@app.route('/')
def index():
    return render_template('home.html')


# @app.route('/upload', methods=['GET', 'POST'])
# def upload(image):
#     import numpy as np
#     model = load_model('models/model_weights.h5')
#     img_path = image
#     img = keras.preprocessing.image.load_img(img_path, target_size=(224,224))
#     img_array = keras.preprocessing.image.img_to_array(img)
#     expanded_img_array = np.expand_dims(img_array, axis=0)
#     preprocessed_img = expanded_img_array / 255. # Preprocess the image
#     prediction = model.predict(preprocessed_img)
#     print(prediction)

#     return render_template('upload.html')
@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():

    if request.method == 'POST':

        if request.files:
            import keras
            from keras.models import load_model
            import numpy as np
            model = load_model('/Volumes/T5_500G/Capstone/v2/flask/models/model_weights.h5')

            image2 = request.files['image']
            image = os.path.join(app.config['IMAGE_UPLOADS'], image2.filename)
            image2.save(image)
            img_path = image
            img = keras.preprocessing.image.load_img(img_path, target_size=(224,224))
            img_array = keras.preprocessing.image.img_to_array(img)
            expanded_img_array = np.expand_dims(img_array, axis=0)
            preprocessed_img = expanded_img_array / 255. # Preprocess the image
            prediction = model.predict(preprocessed_img)
            print(prediction)
            # print(image2)
            #print (request.url)

            # return redirect(request.url)


    return render_template('/upload_image.html')

# @app.route('/doughnut_chart')
# def doughnut_chart():
#     return render_template('doughnut_chart.html')

@app.route('/line_chart')
def line_chart():
    return render_template('line_chart.html')




if __name__ == '__main__':
    app.run(debug=True)