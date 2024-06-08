import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.title("Naive Bayes Classifier for Document Classification")

    # Sample data
    data = {
        'Text': ["great movie", "bad acting", "awesome film", "poor direction", "loved it"],
        'Sentiment': [1, 0, 1, 0, 1]  # 1: Positive, 0: Negative
    }

    df = pd.DataFrame(data)

    st.write("The sample dataset is:")
    st.write(df)

    # Select text and target columns
    text_column = st.selectbox("Select Text Column", df.columns.tolist())
    target_column = st.selectbox("Select Target Column", df.columns.tolist(), index=len(df.columns)-1)

    if text_column != target_column:
        X = df[text_column]
        y = df[target_column]

        # Split data into train and test sets
        split_ratio = st.slider("Training Set Ratio", 0.1, 0.9, 0.7)
        split_index = int(split_ratio * len(df))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Train Naive Bayes model
        prior_probabilities = {}
        likelihoods = {}

        # Calculate prior probabilities
        for label in np.unique(y_train):
            prior_probabilities[label] = np.sum(y_train == label) / len(y_train)

        # Calculate likelihoods
        for label in np.unique(y_train):
            label_indices = np.where(y_train == label)[0]
            label_X = X_train.iloc[label_indices]
            likelihoods[label] = {}
            for word in label_X.str.split():
                likelihoods[label][word] = (label_X.str.split().apply(lambda x: x.count(word)) + 1) / (len(label_X) + len(X_train.unique()))

        # Make predictions
        y_pred = []
        for x in X_test:
            label_scores = {}
            for label in np.unique(y_train):
                score = np.log(prior_probabilities[label])
                for word in x.split():
                    if word in likelihoods[label]:
                        score += np.log(likelihoods[label][word])
                label_scores[label] = score
            y_pred.append(max(label_scores, key=label_scores.get))

        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        precision = np.sum((y_pred == y_test) & (y_test == 1)) / np.sum(y_pred == 1)
        recall = np.sum((y_pred == y_test) & (y_test == 1)) / np.sum(y_test == 1)

        st.write("\nModel Performance:")
        st.write("Accuracy:", accuracy)
        st.write("Precision:", precision)
        st.write("Recall:", recall)

if __name__ == "__main__":
    main()

