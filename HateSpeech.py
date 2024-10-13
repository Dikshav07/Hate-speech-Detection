import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string

nltk.download('stopwords')

app = Flask(__name__)

stemmer = SnowballStemmer("english")

# Set of stopwords in English
stopword = set(stopwords.words("english"))

# Read the Twitter dataset
df = pd.read_csv("C:\\Users\\tanis\\Downloads\\twitter_data.csv")

# Labeling the class for Hate, Offensive, and No Hate and Offensive Language
df['labels'] = df['class'].map({0: "Hate Speech Detected", 1: "Offensive language detected", 2: "No Hate and Offensive Speech"})
df = df[['tweet', 'labels']]

# Function to clean text
def clean(text):
    text = str(text).lower()  # Convert text to lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove text within square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newline characters
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing digits
    text = [word for word in text.split(' ') if word not in stopword]  # Remove stopwords
    text = " ".join(text)  # Join the list of words back into a string
    text = [stemmer.stem(word) for word in text.split(' ')]  # Apply stemming
    text = " ".join(text)  # Join the list of stemmed words back into a string
    return text

# Clean the tweet data
df["tweet"] = df["tweet"].apply(clean)
df = df.dropna()

# Convert Tweet and Label columns of the DataFrame to numpy arrays
x = np.array(df["tweet"])
y = np.array(df["labels"])

# Convert text data to numerical data (CountVectorizer)
cv = CountVectorizer()
x = cv.fit_transform(x)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        # Clean the input message
        cleaned_message = clean(message)
        # Transform the cleaned message using CountVectorizer
        message_transformed = cv.transform([cleaned_message]).toarray()
        # Predict the label
        prediction = clf.predict(message_transformed)
        # Prepare the predicted label for rendering in HTML
        prediction_label = prediction[0]
        return render_template('index.html', prediction=prediction_label)

if __name__ == "__main__":
    app.run(debug=True)
