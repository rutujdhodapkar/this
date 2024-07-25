import gradio as gr
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
data_path = 'Emotion_final.csv'
data = pd.read_csv(data_path)

# Assuming the dataset has 'text' and 'emotion' columns
X = data['Text']
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

# Save the model and vectorizer
joblib.dump(logistic_regression_model, 'logistic_regression_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Load the model and vectorizer
logistic_regression_model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to predict emotion from text using Logistic Regression
def predict_emotion_logistic(text):
    text_vec = vectorizer.transform([text])
    prediction = logistic_regression_model.predict(text_vec)
    return prediction[0]

# Gradio interface
def predict_emotion_interface(text):
    predicted_emotion = predict_emotion_logistic(text)
    return predicted_emotion

iface = gr.Interface(
    fn=predict_emotion_interface,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Enter text here..."),
    outputs="text",
    title="Emotion Prediction using Logistic Regression",
    description="Enter text to predict the emotion using a Logistic Regression model.",
)

if __name__ == "__main__":
    iface.launch()
