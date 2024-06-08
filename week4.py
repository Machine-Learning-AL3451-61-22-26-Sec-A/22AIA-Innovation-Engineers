import streamlit as st
import pandas as pd
import numpy as np

def label_encode(data):
    unique_values = np.unique(data)
    label_map = {val: idx for idx, val in enumerate(unique_values)}
    encoded_data = [label_map[val] for val in data]
    return encoded_data, label_map

def main():
    st.title("Naive Bayes Classifier for Tennis Play Prediction")

    # Tennis Play dataset
    data = {
        'Outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
        'Temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'],
        'Humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'],
        'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
        'PlayTennis': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
    }
    df = pd.DataFrame(data)

    st.write("The first 5 rows of the dataset are:")
    st.write(df.head())

    # Select target
    target = st.selectbox("Select Target", df.columns.tolist(), index=len(df.columns)-1)

    if target:
        X = df.drop(columns=[target]).copy()
        y, _ = label_encode(df[target])
        y = np.array(y)

        # Train-test split
        split_ratio = st.slider("Training Set Ratio", 0.1, 0.9, 0.7)
        split_index = int(split_ratio * len(df))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Train Naive Bayes model
        prior_probabilities = {}
        for label in np.unique(y_train):
            prior_probabilities[label] = np.sum(y_train == label) / len(y_train)

        likelihoods = {}
        for feature in X_train.columns:
            likelihoods[feature] = {}
            for label in np.unique(y_train):
                label_indices = np.where(y_train == label)[0]
                feature_values = X_train.iloc[label_indices][feature]
                value_counts = np.bincount(feature_values)
                total_counts = np.sum(value_counts)
                likelihoods[feature][label] = {value: count / total_counts for value, count in enumerate(value_counts)}

        # Predict using Naive Bayes
        def predict(X):
            predictions = []
            for idx, row in X.iterrows():
                posterior_probabilities = {label: prior_probabilities[label] for label in np.unique(y_train)}
                for feature, value in row.items():
                    if value in likelihoods[feature][label]:
                        posterior_probabilities[label] *= likelihoods[feature][label][value]
                    else:
                        posterior_probabilities[label] *= 1e-6
                predictions.append(max(posterior_probabilities, key=posterior_probabilities.get))
            return predictions

        # Evaluate accuracy
        y_pred = predict(X_test)
        accuracy = np.mean(y_pred == y_test)

        st.write("\nModel Accuracy:", accuracy)

if __name__ == "__main__":
    main()
