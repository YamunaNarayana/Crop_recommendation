from flask import Flask, request, render_template
import numpy as np
import  pandas
import sklearn
import pickle


#importing model
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

#creating flask app

app= Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosphorus'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]

    single_pred = np.array([feature_list])
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)


    Crop_dict={0: 'mungbean', 1: 'papaya', 2: 'coconut', 3: 'apple', 4: 'banana', 5: 'muskmelon', 6: 'pigeonpeas', 7: 'maize',
               8: 'cotton', 9: 'orange', 10: 'grapes', 11: 'pomegranate', 12: 'coffee', 13: 'kidneybeans', 14: 'blackgram',
               15: 'watermelon', 16: 'jute', 17: 'chickpea', 18: 'lentil', 19: 'mothbeans', 20: 'rice', 21: 'mango'}

    if prediction[0] in Crop_dict:
        crop=Crop_dict[prediction[0]]
        result = '{} is the best crop to be cultivated'.format(crop)
    else:
        result = 'Sorry we are not able to recommend a proper crop for this environment'

    return render_template('index.html',result = result)






# python main
if __name__ == '__main__':
    app.run(debug=True)