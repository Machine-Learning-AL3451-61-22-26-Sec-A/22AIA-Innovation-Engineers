import streamlit as st
import numpy as np

# Streamlit app title
st.title("Simple Naive Bayes Classifier")

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

# Tokenize the training data
vocab = set()
for text in training_text:
    vocab.update(text.lower().split())

# Create a dictionary to map words to indices
word_to_idx = {word: i for i, word in enumerate(vocab)}

# Count occurrences of words for each class
word_counts_positive = np.zeros(len(vocab))
word_counts_negative = np.zeros(len(vocab))

for text, label in zip(training_text, training_labels):
    for word in text.lower().split():
        idx = word_to_idx[word]
        if label == 1:
            word_counts_positive[idx] += 1
        else:
            word_counts_negative[idx] += 1

# Compute probabilities
total_positive_words = sum(word_counts_positive)
total_negative_words = sum(word_counts_negative)

prior_positive = sum(training_labels) / len(training_labels)
prior_negative = 1 - prior_positive

# Classify the input text
text_input = st.text_input("Enter text to classify:", "")
if text_input:
    # Tokenize the input text
    input_vector = np.zeros(len(vocab))
    for word in text_input.lower().split():
        if word in word_to_idx:
            input_vector[word_to_idx[word]] += 1
    
    # Calculate likelihoods
    likelihood_positive = np.prod((word_counts_positive + 1) ** input_vector)
    likelihood_negative = np.prod((word_counts_negative + 1) ** input_vector)
    
    # Calculate posteriors
    posterior_positive = likelihood_positive * prior_positive
    posterior_negative = likelihood_negative * prior_negative
    
    # Make prediction
    prediction = 1 if posterior_positive > posterior_negative else 0
    
    # Display the result
    if prediction == 1:
        st.write("Positive sentiment!")
    else:
        st.write("Negative sentiment!")
