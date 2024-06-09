import streamlit as st
import pandas as pd
import numpy as np

# Function to calculate the entropy of a dataset
def entropy(column):
    elements, counts = np.unique(column, return_counts=True)
    return -np.sum([(counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])

# Function to calculate the information gain of a split
def info_gain(data, split_attribute, target_name="class"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute], return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    return total_entropy - weighted_entropy

# ID3 Algorithm
def id3(data, features, target_attribute_name="class"):
    target_values = data[target_attribute_name]
    
    if len(np.unique(target_values)) == 1:
        return np.unique(target_values)[0]
    
    if len(features) == 0:
        return np.unique(target_values)[np.argmax(np.unique(target_values, return_counts=True)[1])]
    
    best_feature = features[np.argmax([info_gain(data, feature, target_attribute_name) for feature in features])]
    tree = {best_feature: {}}
    
    for value in np.unique(data[best_feature]):
        sub_data = data.where(data[best_feature] == value).dropna()
        subtree = id3(sub_data, [feat for feat in features if feat != best_feature], target_attribute_name)
        tree[best_feature][value] = subtree
    
    return tree

# Function to predict a single instance using the decision tree
def predict(query, tree):
    for key in query.keys():
        if key in tree.keys():
            try:
                result = tree[key][query[key]]
            except:
                return 'No'
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result
    return 'No'

# Streamlit app
def main():
    st.write("22AIA-INNOVATIVE ENGINEERS")
    st.title("ID3 Algorithm")

    # Sample data
    data = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
        'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    }
    df = pd.DataFrame(data)

    st.write("### Training Examples")
    st.write(df)

    features = df.columns[:-1]
    tree = id3(df, features, 'PlayTennis')
    
    st.write("### Decision Tree")
    st.json(tree)

    st.write("### Make a Prediction")
    query = {feature: st.selectbox(feature, df[feature].unique()) for feature in features}
    prediction = predict(query, tree)
    st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
