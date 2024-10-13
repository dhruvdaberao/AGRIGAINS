from flask import Flask, request, render_template, Response, jsonify
import numpy as np
import pandas as pd
import pickle, joblib
import ast

# importing model
model = pickle.load(open('./static/model/model.pkl', 'rb'))

sc = pickle.load(open('./static/model/standscaler.pkl', 'rb'))
ms = pickle.load(open('./static/model/minmaxscaler.pkl', 'rb'))
crop_pred = joblib.load("./static/model/crop_predict.pkl")
fertilizer_model = pickle.load(open("./static/model/classifier2.pkl", 'rb'))

# creating flask app
app = Flask(__name__)

def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):

    # Create an array of the input features
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)

    # Ensure all features are numeric as in your training data
    # You likely one-hot encoded 'Area' and 'Item' previously.
    # Use the same preprocessing technique here to transform your input data
    # For one-hot encoding, manually create a new feature vector with the same structure as X_train


    # One way to handle categorical variables is to manually create dummy variables using the known categories from your training set
    # Then ensure that Area and Item are among these categories and add these manual one-hot encodings to your input features
    features_df = pd.DataFrame(features, columns = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item'])
    features_df['Area'] = features_df['Area'].astype(str)
    features_df = pd.get_dummies(features_df, columns=['Item', 'Area'], drop_first=True)
    features_df = features_df.reindex(columns = crop_pred.feature_names_in_, fill_value=0)


    # Make the prediction using predict()
    predicted_yield = crop_pred.predict(features_df).reshape(1, -1)

    return predicted_yield[0]


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/suggestion')
def suggest():
    return render_template("suggestion.html")

@app.route('/services')
def services():
    return render_template("services.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/help')
def help():
    return render_template("help.html")

@app.route('/fertilizer')
def fertilizer():
    return render_template("fertilizer.html")

@app.route('/fertilizer_predict', methods=['POST'])
def fertilizer_predict():
    # Nitrogen = request.form.get('Nitrogen')
    # Potassium = request.form.get('Potassium')
    # Phosphorous = request.form.get('Phosphorous')
    data = ast.literal_eval(request.get_data().decode("utf-8"))
    Nitrogen = data["nitrogen"]
    Potassium = data["potassium"]
    Phosphorous = data["phosphorous"]

    # prediction
    result = fertilizer_model.predict(np.array([[Nitrogen, Potassium, Phosphorous]]))

    if result[0] == 0:
        result = 'TEN-TWENTY SIX-TWENTY SIX'
    elif result[0] == 1:
        result = 'Fourteen-Thirty Five-Fourteen'
    elif result[0] == 2:
        result = 'Seventeen-Seventeen-Seventeen'
    elif result[0] == 3:
        result = 'TWENTY-TWENTY'
    elif result[0] == 4:
        result = 'TWENTY EIGHT-TWENTY EIGHT'
    elif result[0] == 5:
        result = 'DAP'
    else:
        result = 'UREA'

    print(result)
    return jsonify({"result": result})

@app.route("/predict", methods=['POST'])
def predict():
    # print(request)
    # N = request.form['Nitrogen']
    # P = request.form['Phosporus']
    # K = request.form['Potassium']
    # temp = request.form['Temperature']
    # humidity = request.form['Humidity']
    # ph = request.form['Ph']
    # rainfall = request.form['Rainfall']
    
    data = ast.literal_eval(request.get_data().decode("utf-8"))
    N = data["N"]
    P = data["P"]
    K = data["K"]
    temp = data["temp"]
    humidity = data["humidity"]
    ph = data["ph"]
    rainfall = data["rainfall"]
    

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Chickpea", 4: "Kidneybeans", 5: "Pigeonpeas", 6: "Mothbeans", 
        7: "Mungbean", 8: "Blackgram", 9: "Lentil", 10: "Pomegranate", 11: "Banana", 12: "Mango", 
        13: "Grapes", 14: "Watermelon", 15: "Muskmelon", 16: "Apple", 17: "Orange", 18: "Papaya", 
        19: "Coconut", 20: "Cotton", 21: "Jute", 22: "Coffee"
    }

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        print(crop)
        result = "{} is the best crop to be cultivated right there".format(crop)
   
    else:
       result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    # return render_template('suggestion.html', result=result)
    print(result)
    return jsonify(result=result, imgID=crop)

@app.route("/predictYield")
def predictYield():
    return render_template("predict.html")

@app.route("/pYield", methods=['POST'])
def pYield():
    data = ast.literal_eval(request.get_data().decode("utf-8"))
    year = data["year"]
    average_rainfall = data["average_rainfall_mm_per_year"]
    pesticides = data["pesticides_tonnes"]
    avg_temp = data["avg_temp"]
    area = data["area"]
    item = data["item"]

    result = prediction(Year=year, average_rain_fall_mm_per_year=average_rainfall, pesticides_tonnes=pesticides, avg_temp=avg_temp, Area=area, Item=item)
    print(result)
    return jsonify({"result": result[0]})

# python main
if __name__ == "__main__":
    app.run("127.0.0.1", 5500, debug=True)
