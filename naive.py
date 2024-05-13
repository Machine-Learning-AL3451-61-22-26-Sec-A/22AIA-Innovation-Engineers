import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Generate a simple dataset of movie reviews and sentiments
reviews = [
    "This movie was great!",
    "I really liked this movie.",
    "Awful movie, I hated it.",
    "The worst movie ever made.",
    "Not bad, but could have been better."
]
sentiments = ["positive", "positive", "negative", "negative", "negative"]

# Text preprocessing
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)
y = sentiments

# Training the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, y)

# Streamlit UI
st.title("Movie Review Sentiment Classifier")

review_text = st.text_input("Enter your movie review:")

if review_text:
    review_text = [review_text]
    review_vector = vectorizer.transform(review_text)
    prediction = clf.predict(review_vector)
    if prediction[0] == 'positive':
        st.write("Prediction: Positive 😊")
    else:
        st.write("Prediction: Negative 😞")
