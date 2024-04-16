import numpy as np

class Node:
    def __init__(self, attribute=None, value=None, leaf_class=None):
        self.attribute = attribute  # Attribute to split on
        self.value = value          # Value of the attribute
        self.leaf_class = leaf_class  # Class label if it's a leaf node
        self.children = {}          # Dictionary to hold child nodes

def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def information_gain(X, y, attribute):
    entropy_parent = entropy(y)
    values, counts = np.unique(X[:, attribute], return_counts=True)
    weighted_entropy_children = np.sum((counts/np.sum(counts)) * entropy(y[X[:, attribute] == value]) for value, count in zip(values, counts))
    information_gain = entropy_parent - weighted_entropy_children
    return information_gain

def id3(X, y, attributes):
    if len(np.unique(y)) == 1:  # If all instances have the same class
        return Node(leaf_class=y[0])
    if len(attributes) == 0:  # If there are no more attributes to split on
        return Node(leaf_class=np.argmax(np.bincount(y)))  # Return the majority class

    best_attribute = max(attributes, key=lambda attr: information_gain(X, y, attr))
    node = Node(attribute=best_attribute)

    values = np.unique(X[:, best_attribute])
    for value in values:
        X_subset = X[X[:, best_attribute] == value]
        y_subset = y[X[:, best_attribute] == value]
        if len(X_subset) == 0:  # If subset is empty, return majority class
            node.children[value] = Node(leaf_class=np.argmax(np.bincount(y)))
        else:
            remaining_attributes = [attr for attr in attributes if attr != best_attribute]
            node.children[value] = id3(X_subset, y_subset, remaining_attributes)

    return node

# Example data
X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y = np.array([1, 0, 1, 0])
attributes = [0, 1]

# Building the decision tree
root = id3(X, y, attributes)

# Output
print("Decision tree:")
def print_tree(node, depth=0):
    if node.leaf_class is not None:
        print("  " * depth + "Class:", node.leaf_class)
    else:
        print("  " * depth + "Attribute:", node.attribute)
        for value, child in node.children.items():
            print("  " * (depth + 1) + "Value:", value)
            print_tree(child, depth + 2)

print_tree(root)
