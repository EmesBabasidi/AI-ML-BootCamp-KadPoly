from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load your trained model
model = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = [message]
    data_reshaped = vectorizer.transform(data)  # Transform the message using the vectorizer
    prediction = model.predict(data_reshaped)  # Pass the transformed data to the model    
    return render_template('index.html', prediction_text=f'This message is {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)
