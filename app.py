import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Set up page configuration
st.set_page_config(page_title="Sentiment Analysis", layout="centered", initial_sidebar_state="collapsed")

# Load the pre-trained model
@st.cache_resource
def load_sentiment_model():
    try:
        return load_model('model.keras', compile=False)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model = load_sentiment_model()


# Load the tokenizer from the pickle file
with open('./tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


# Main function to predict sentiment
def predict_sentiment(text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=200)  # Ensure padding length matches the model

    # Predict the sentiment
    prediction = model.predict(padded_sequences)
    sentiment = 'Positive' if prediction >= 0.5 else 'Negative'
    return sentiment

# Streamlit UI
st.title("Sentiment Analysis Web App")
st.caption("Analyze the sentiment of text as Positive or Negative using a deep learning model.")

# Add a responsive image
st.image("bg.png", caption="Sentiment Analysis", use_container_width=True)

# Input area for user text
user_input = st.text_area("Enter your text here:", placeholder="Type something meaningful...")

# Button to trigger prediction
if st.button("Predict Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)

        # Dynamically style the output based on sentiment
        if sentiment == "Positive":
            st.balloons()  # Celebrate positive sentiment
            st.success(f"😊 The sentiment of the text is: **{sentiment}**")
        else:
            st.error(f"😞 The sentiment of the text is: **{sentiment}**")
    else:
        st.warning("Please enter some text for prediction.")

# Footer message
st.markdown("---")
st.markdown("**Note:** This web app is optimized for both desktop and mobile devices. Remeber the model is trained on IMDB 50K movies dataset so type accordingly although It can perform for other tasks too.")
