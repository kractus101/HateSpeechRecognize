# Download and Import needed packages and libraries
import streamlit as st  
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve,roc_auc_score, auc,accuracy_score,precision_score,recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression
import re
import nltk
nltk.download("stopwords")
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string

stopword = set(stopwords.words('english')) # set stopwords that needs to be removed while cleaning

data = pd.read_csv("C:/Users/emyes/OneDrive/Desktop/AMOD_Major_Research_Paper/Data/labeled_data.csv") # Read the dataset

data["labels"] = data["class"].map({0: "Hate Speech",    # mapping each class of the dataset
                                    1: "Offensive Language",
                                    2: "No Hate and Offensive"})
data = data[["tweet", "labels"]]  # only selecting the tweet and labels column 


def clean(text):       # cleaning the text data
    text = str(text).lower()  # convert to lower text
    text = re.sub('\[.*?\]', '', text)  #remove special characters specified
    text = re.sub('https?://\S+|www\.\S+', '', text) #remove URL and website addresses
    text = re.sub('<.*?>+', '', text)  #Remove HTML tags
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation characters
    text = re.sub('\n', '', text)  # Remove newline characters
    text = re.sub('\w*\d\w*', '', text) # Remove words containing digits
    text = [word for word in text.split(' ') if word not in stopword]  #remove stop words
    text = " ".join(text)  #join the list of words into a single string
    text = [stemmer.stem(word) for word in text.split(' ')]  #apply stemming
    text = " ".join(text)  #Join the list of stemmed words into a single string
    return text


data["tweet"] = data["tweet"].apply(clean)  # apply the cleaning on tweet column

x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()  # Initialize a CountVectorizer for converting text data into a bag-of-words representation
X = cv.fit_transform(x)  # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) # Split the dataset into training and testing sets

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

clf.fit(X_train, y_train)  # Train the selected classifier on the training data
accuracy = clf.score(X_test, y_test) # Evaluate the accuracy of the classifier on the test data
st.write(f"Accuracy: {accuracy * 100:.2f}%")

def hate_speech_detection():   # Define a function for hate speech detection using the trained classifier
    st.title("Hate Speech Detection")
    user = st.text_area("Enter any Tweet: ")
    if len(user) < 1:
        st.write("  ")
    else:            # Transform the user's input using the same CountVectorizer
        sample = user
        data = cv.transform([sample]).toarray()
        prediction = clf.predict(data)    # Make a prediction using the trained classifier
        st.title(f"Predicted Label: {prediction[0]}")

hate_speech_detection()  # Call the hate_speech_detection function

# Train three additional classifiers for comparison

lr_classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
lr_classifier.fit(X_train, y_train)
lr_pred = lr_classifier.predict(X_test)

svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
svc_pred = svm_classifier.predict(X_test)

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)

svc_cm = confusion_matrix(y_test, svc_pred)
dt_cm = confusion_matrix(y_test, dt_pred)
lr_cm = confusion_matrix(y_test, lr_pred)

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Confusion Matrix for Logistic Regression
sns.heatmap(lr_cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
axes[0].set_title("Logistic Regression")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

# Confusion Matrix for Decision Tree
sns.heatmap(dt_cm, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[1])
axes[1].set_title("Decision Tree")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

# Confusion Matrix for Another Model (you can replace this with your third model)
sns.heatmap(svc_cm, annot=True, fmt="d", cmap="Reds", cbar=False, ax=axes[2])
axes[2].set_title("SVC Model")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("True")

plt.tight_layout()
plt.show()


# Plot confusion matrix for Logistic Regression
plt.figure(figsize=(10, 6))


plt.subplot(1, 2, 1)
svc_cm = confusion_matrix(y_test, svc_pred)
sns.heatmap(svc_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Support Vector Classifier")
plt.xlabel("Predicted")
plt.ylabel("True")

# Plot confusion matrix for Decision Tree
plt.subplot(1, 2, 2)
dt_cm = confusion_matrix(y_test, dt_pred)
sns.heatmap(dt_cm, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

#Plot confusion matrix for SVM
plt.subplot(1, 2, 3)
svc_cm = confusion_matrix(y_test, svc_pred)
sns.heatmap(svc_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Support Vector Classifier")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()
plt.show()


############# ACCURACY #############


# Calculate accuracy score
lr_accuracy = accuracy_score(y_test, lr_pred)
svc_accuracy = accuracy_score(y_test, svc_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)


# Print accuracy score
print(f"Logistic Regression accuracy: {lr_accuracy * 100:.2f}%")
print(f"SVC accuracy: {svc_accuracy * 100:.2f}%")
print(f"Decision Tree accuracy: {dt_accuracy * 100:.2f}%")

lr_precision = precision_score(y_test, lr_pred)

############# PRECISION #############

# Calculate precision score
lr_per_class = precision_score(y_test, lr_pred, average='weighted')
svc_precision = precision_score(y_test, svc_pred, average='weighted')
dt_precision = precision_score(y_test, dt_pred, average='weighted')

# Print precision score
print(f"LR precision: {lr_per_class * 100:.2f}%")
print(f"SVC precision: {svc_precision* 100:.2f}%")
print(f"Decision Tree precision: {dt_precision* 100:.2f}%")

# ############# RECALL #############

# Calculate recall score
lr_recall = recall_score(y_test, lr_pred, average='weighted')
svc_recall = recall_score(y_test, svc_pred, average='weighted')
dt_recall = recall_score(y_test, dt_pred, average='weighted')

# Print recall score
print(f"LR recall: {lr_recall * 100:.2f}%")
print(f"SVC recall: {svc_recall* 100:.2f}%")
print(f"Decision Tree recall: {dt_recall* 100:.2f}%")

############### F1-score ############

# Calculate F1 score
lr_f1 = f1_score(y_test, lr_pred, average='weighted')
svc_f1 = f1_score(y_test, svc_pred, average='weighted')
dt_f1 = f1_score(y_test, dt_pred, average='weighted')

# Print F1 score
print(f'LR F1 Score: {lr_f1* 100:.2f}%')
print(f'SVC F1 Score: {svc_f1* 100:.2f}%')
print(f'Decision Tree F1 Score: {dt_f1* 100:.2f}%')


