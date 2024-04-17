import streamlit as st
from collections import Counter
import math

class Node:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute
        self.label = label
        self.children = {}

def entropy(data):
    n = len(data)
    label_counts = Counter(data)
    entropy = 0
    for label in label_counts:
        p_label = label_counts[label] / n
        entropy -= p_label * math.log2(p_label)
    return entropy

def information_gain(data, attribute_index):
    n = len(data)
    attribute_values = set([example[attribute_index] for example in data])
    attribute_entropy = 0
    for value in attribute_values:
        subset = [example[-1] for example in data if example[attribute_index] == value]
        attribute_entropy += (len(subset) / n) * entropy(subset)
    return entropy([example[-1] for example in data]) - attribute_entropy

def id3(data, attributes):
    labels = [example[-1] for example in data]
    if len(set(labels)) == 1:
        return Node(label=labels[0])
    if len(attributes) == 0:
        return Node(label=Counter(labels).most_common(1)[0][0])
    best_attribute_index = max(range(len(attributes)), key=lambda i: information_gain(data, i))
    best_attribute = attributes[best_attribute_index]
    node = Node(attribute=best_attribute)
    attribute_values = set([example[best_attribute_index] for example in data])
    for value in attribute_values:
        subset = [example for example in data if example[best_attribute_index] == value]
        if len(subset) == 0:
            node.children[value] = Node(label=Counter(labels).most_common(1)[0][0])
        else:
            child_attributes = attributes[:best_attribute_index] + attributes[best_attribute_index + 1:]
            node.children[value] = id3(subset, child_attributes)
    return node

def print_tree(node, depth=0):
    if node.label is not None:
        st.write("  " * depth, "Predict:", node.label)
    else:
        st.write("  " * depth, "Attribute:", node.attribute)
        for value, child_node in node.children.items():
            st.write("  " * (depth + 1), "Value:", value)
            print_tree(child_node, depth + 2)

def main():
    st.title("ID3 Algorithm")

    # Sample dataset
    sample_dataset = [
        ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Y'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Y'],
        ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'N'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Y']
    ]

    st.subheader("Sample Dataset")
    for row in sample_dataset:
        st.write(row)

    attributes = [f"Attribute {i}" for i in range(len(sample_dataset[0]) - 1)]
    decision_tree = id3(sample_dataset, attributes)

    st.subheader("Decision Tree")
    print_tree(decision_tree)

if __name__ == "__main__":
    main()
