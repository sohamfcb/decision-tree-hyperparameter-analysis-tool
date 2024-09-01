import streamlit as st

st.set_page_config(page_title='Home')

st.title('Welcome to our Decision Tree Hyperparameter Analysis Tool!')
st.image('Screenshot 2024-09-01 152533.png',width=600)

st.write()

st.markdown("<h3><u>A Brief Overview of Decision Trees</u></h3>", unsafe_allow_html=True)

st.write('''
    Decision Trees are a popular and intuitive machine learning algorithm used for classification and regression tasks. They work by recursively splitting the data into subsets based on feature values, creating a tree-like model of decisions and their possible consequences.
Structure

Nodes: Represent decisions or tests on features. Each internal node is a test on a feature, and each leaf node represents a class label (for classification) or a continuous value (for regression).
Branches: Represent the outcome of the tests and connect nodes in the tree.
Root Node: The topmost node from which all decisions start. It represents the entire dataset and is split into sub-nodes.

How They Work

Splitting: At each node, the dataset is divided into subsets based on the values of a feature. The goal is to find splits that lead to the purest possible subsets, meaning subsets where the data points are mostly of one class (for classification) or close to each other (for regression).
Criteria: Decision trees use criteria such as Gini impurity, entropy, or mean squared error (for regression) to decide the best feature and value for splitting.
Pruning: To prevent overfitting, decision trees can be pruned, which involves removing nodes that provide little predictive power to simplify the model.

Advantages

Interpretability: Decision trees are easy to interpret and visualize. The decision-making process can be followed from the root to the leaves.
Handling Non-linearity: They do not require assumptions about the relationship between features and the target variable, making them capable of handling non-linear relationships.

Disadvantages

Overfitting: They can easily overfit the training data, especially with very deep trees. Techniques like pruning and setting a maximum depth can help mitigate this.
Instability: Small changes in the data can lead to different splits and thus a different tree structure.

Applications

Classification: Used in scenarios where the goal is to classify data into categories, such as in medical diagnoses, customer segmentation, and spam detection.
Regression: Used to predict continuous values, such as predicting house prices or stock prices.
''')