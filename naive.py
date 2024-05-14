import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Naive Bayes Classifier
class NaiveBayesClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.classifier = MultinomialNB()

    def train(self, X, y):
        X_vec = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_vec, y)

    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.classifier.predict(X_vec)

# Streamlit UI
def main():
    st.title("Text Classifier")
    st.write("This is a simple Naive Bayes classifier for text classification.")

    # Train the classifier
    X_train = ["I love to play football", "It's raining outside", "The sun is shining"]
    y_train = ["sports", "weather", "weather"]
    classifier = NaiveBayesClassifier()
    classifier.train(X_train, y_train)

    text_input = st.text_input("Enter text to classify:", "")

    if st.button("Classify"):
        if text_input:
            predicted_class = classifier.predict([text_input])[0]
            st.write("Predicted Class:", predicted_class)
        else:
            st.write("Please enter some text.")

if __name__ == "__main__":
    main()
