IMDB Movie Review Sentiment Analysis

Overview

This project involves sentiment analysis on the IMDB Movie dataset, which consists of 50,000 movie reviews labeled as positive or negative. The dataset is balanced, ensuring that evaluation metrics such as accuracy and F1-score are representative of model performance.

Techniques Explored

I systematically experimented with various machine learning and deep learning techniques to determine the best-performing model.

1. Traditional Machine Learning Approaches

Performed feature extraction using:

Bag of Words (BoW)

TF-IDF Vectorizer

Then applied the following models:

Logistic Regression

Support Vector Machine (SVM)

Random Forest Classifier

Multinomial Naïve Bayes

2. Artificial Neural Networks (ANNs)

After applying BoW and TF-IDF feature extraction, I trained ANN models using Keras and tuned hyperparameters with Keras Tuner.

3. Sequential Deep Learning Models

Using Keras Preprocessing, I implemented:

SimpleRNN

LSTM (Long Short-Term Memory)

Final Model and Performance

After extensive experimentation, I found that the LSTM model performed the best with:

Final Accuracy: 88.87%

F1-Score: Similar to accuracy due to the balanced dataset.

Model Export

To deploy the model, I exported:

The trained LSTM model

The Tokenizer used for text preprocessing

Sentiment Analysis Web App

This is a simple Sentiment Analysis Web App built with Streamlit and a pre-trained deep learning model. The app predicts whether a given text has a Positive or Negative sentiment.

Features

Real-time Sentiment Prediction: Enter any text, and the app will analyze its sentiment.

Interactive UI: Clean and user-friendly interface built with Streamlit.

Responsive Design: Works seamlessly on both mobile and desktop devices.

How It Works

The input text is preprocessed using a Tokenizer.

The text is padded to ensure uniform input size for the deep learning model.

A TensorFlow model predicts the sentiment based on the processed input.

Setup and Deployment

Requirements

Python 3.8 or later

Libraries listed in requirements.txt

Steps to Run Locally

Clone the repository:

git clone https://github.com/<your-username>/sentiment-analysis-app.git
cd sentiment-analysis-app

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py

Open the local server URL in your browser to interact with the app.

Deployment

The app is deployed using Streamlit Community Cloud. You can access the live app here: Sentiment Analysis Web App

File Structure

Sentiment_Analysis_App/
│
├── app.py                    # Main Streamlit application script
├── bg.png                    # Background image
├── model.keras               # Pre-trained deep learning model
├── tokenizer.pkl             # Tokenizer used for preprocessing
├── requirements.txt          # Dependencies for the app

Conclusion

Through this project, I explored multiple machine learning and deep learning techniques for sentiment classification. The LSTM model, leveraging sequential text processing, achieved the best performance, making it the most suitable model for this task.
Tackled overfitting issue:
<img width="472" alt="model1_history" src="https://github.com/user-attachments/assets/32fec655-3bf1-4857-8eb7-2a8a5e2a5151" />

