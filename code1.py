import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression
import re
import nltk

# nltk.download("stopwords")
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string

stopword = set(stopwords.words('english'))

data = pd.read_csv("labeled_data.csv")

data["labels"] = data["class"].map({0: "Hate Speech",
                                    1: "Offensive Language",
                                    2: "No Hate and Offensive"})
data = data[["tweet", "labels"]]


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text


data["tweet"] = data["tweet"].apply(clean)

x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x)  # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create a dropdown menu to select the classification method
classification_method = st.selectbox("Select Classification Method", ["Decision Tree", "SVM", "Logistic Regression"])

if classification_method == "Decision Tree":
    clf = DecisionTreeClassifier()  # Use decision tree 
elif classification_method == "SVM":
    clf = SVC()  # Use SVM
elif classification_method == "Logistic Regression":
    clf = LogisticRegression()  # Use Logistic Regression
else:
    st.error("Invalid classification method selected.")

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
st.write(f"Accuracy: {accuracy * 100:.2f}%")

def hate_speech_detection():
    st.title("Hate Speech Detection")
    user = st.text_area("Enter any Tweet: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        prediction = clf.predict(data)
        st.title(f"Predicted Label: {prediction[0]}")

hate_speech_detection()
