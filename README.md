# Hate-speech-Detection
![Static Badge](https://img.shields.io/badge/Python-3.8-blue)
![Static Badge](https://img.shields.io/badge/Framwork-Flask-red)
![Static Badge](https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-Purple)
![Static Badge](https://img.shields.io/badge/API-TMBD-yellow)
<br>
Hate speech has become a growing concern in today’s online world. Identifying and combating hate speech is important to maintain healthy and respectful discussions on various platforms. This Hate Speech Detection project is designed to help identify and classify text as hate speech or non-hate speech using machine learning algorithms.

The project consists of a Flask backend with a web frontend to make the model accessible and user-friendly.
# Features
1. **Text Classification**: Detects whether the input text contains hate speech or not.
2. **Machine Learning Model**: Trained on labeled data for hate speech detection.
3. **Web Interface**: A simple and interactive UI for users to test the model.
4. **Data Visualization**: Graphical representation of predictions and results.
5. **Responsive Design**: Mobile-friendly interface.
6. **Light-weight and Efficient**: Fast processing of text with minimal delay.

### Tech Stack
- **Frontend**:
  - HTML
  - CSS
  - JavaScript

- **Backend**:
  - Flask (Python)

- **Machine Learning**:
  - Scikit-learn (Python)
  - Natural Language Processing (NLP)
  - Pandas, Numpy

- **Deployment**:
  - Flask server for backend
  - Streamlit for quick deployment (optional)

# How to run the project?
1. Clone or download this repository to your local machine.
2. Install all the libraries mentioned in the requirements.txt file with the command pip install -r requirements.txt
3. Open your terminal/command prompt from your project directory and run the file main.py by executing the command python app.py
4. Go to your browser and type http://127.0.0.1:5000/ in the address bar.
5. And That's it!!

# Architecture
![Screenshot (32)](https://github.com/user-attachments/assets/3d9332ad-1a91-4150-8ac6-d797332562f7)

# Model Training
The machine learning model used in this project is based on natural language processing (NLP) techniques. The following steps were used to train the model:

1. **Data Preprocessing**: Cleaning and preparing the dataset.
    -Tokenization
    -Removal of stop words
    -Lemmatization and stemming
2. **Vectorization**: Using CountVectorizer and TF-IDF to convert text to numerical format.
3. **Model**: Various classification models such as Logistic Regression, Naive Bayes, and Support Vector Machine (SVM) were evaluated.
4. **Evaluation**: The best-performing model was selected based on accuracy, precision, recall, and F1-score.

# Code for Model Training:
You can find the model training script in the train_model.py file. It handles data preprocessing, training, and saving the trained model.
