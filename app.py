from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    vectorizer, model = pickle.load(model_file)  # Load as a tuple

def get_prediction_label(prediction):
    label_mapping = {
        0: "No Hate and Offensive Speech",
        1: "Hate Speech Detected"
    }
    return label_mapping.get(prediction, "Unknown")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        
        data = vectorizer.transform([message])
        
        prediction = model.predict(data)[0]
       
        prediction_label = get_prediction_label(prediction)
        return render_template('index.html', prediction=prediction_label)

if __name__ == "__main__":
    app.run(debug=True)
