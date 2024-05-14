import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Streamlit app title
st.title("Simple Naive Bayes Classifier")

# Text input for user to input text
text_input = st.text_input("Enter text to classify:", "")

# Example training data
training_text = [
    "I love this sandwich.",
    "This is an amazing place!",
    "I feel very good about these beers.",
    "This is my best work.",
    "What an awesome view",
    "I do not like this restaurant",
    "I am tired of this stuff.",
    "I can't deal with this",
    "He is my sworn enemy!",
    "My boss is horrible."
]
training_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1 for positive sentiment, 0 for negative sentiment

# Transform the training data into a document-term matrix
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_text)

# Initialize and train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, training_labels)

# Classify the input text
if text_input:
    # Transform the input text into a document-term matrix
    input_text = vectorizer.transform([text_input])
    
    # Make predictions
    prediction = clf.predict(input_text)
    
    # Display the result
    if prediction[0] == 1:
        st.write("Positive sentiment!")
    else:
        st.write("Negative sentiment!")
