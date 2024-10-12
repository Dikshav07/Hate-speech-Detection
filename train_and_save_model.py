import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

# Example training data
X_train = ["I love this!", "I hate you!", "You are amazing", "You are ugly"]
y_train = [0, 1, 0, 1]  

vectorizer = CountVectorizer(max_features=5000, stop_words='english')
model = DecisionTreeClassifier()

X_train_vectorized = vectorizer.fit_transform(X_train)
model.fit(X_train_vectorized, y_train)

with open('model.pkl', 'wb') as model_file:
    pickle.dump((vectorizer, model), model_file)
