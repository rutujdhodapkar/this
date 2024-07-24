#logistic

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data_path = r'C:\Users\rutuj\OneDrive\Documents\OneDrive\Desktop\emoemo.csv'
data = pd.read_csv(data_path)

# Assuming the dataset has 'text' and 'emotion' columns
X = data['text']
y = data['Emotion']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize the Logistic Regression model
logistic_regression_model = LogisticRegression(max_iter=1000)

# Train the model
logistic_regression_model.fit(X_train_vec, y_train)

# Make predictions
y_pred = logistic_regression_model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Function to predict emotion from text using Logistic Regression
def predict_emotion_logistic(text):
    text_vec = vectorizer.transform([text])
    prediction = logistic_regression_model.predict(text_vec)
    return prediction[0]

# Example usage
example_text = "I don't like this"
predicted_emotion = predict_emotion_logistic(example_text)
print(f'Predicted Emotion using Logistic Regression: {predicted_emotion}')

# Streamlit app
st.title('Emotion Prediction using Logistic Regression')

# Display the dataset
st.subheader('Dataset')
st.write(data.head())

# Input text for emotion prediction
st.subheader('Predict Emotion from Text')
input_text = st.text_input('Enter text:')

if st.button('Predict'):
    if input_text:
        predicted_emotion = predict_emotion_logistic(input_text)
        st.write(f'Predicted Emotion: {predicted_emotion}')
    else:
        st.write('Please enter some text to predict the emotion.')