import streamlit as st
import pandas as pd
import numpy as np

# Sentiment Analysis Dataset
data = {
    "text": ["I love this product", "This is amazing", "Awesome experience", "Not good", "Poor quality"],
    "sentiment": ["positive", "positive", "positive", "negative", "negative"]
}
df = pd.DataFrame(data)

# Vectorize text data
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['sentiment']

# Train Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X, y)

# Streamlit app
st.title("Sentiment Analysis with Naive Bayes Classifier")

st.write("### Sample Data")
st.write(df)

user_input = st.text_input("Enter text for sentiment analysis")

if user_input:
    user_input_vectorized = vectorizer.transform([user_input])
    prediction = model.predict(user_input_vectorized)[0]
    st.write(f"Predicted Sentiment: {prediction}")


