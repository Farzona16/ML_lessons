"""1. What is a Decision Tree, and how does it make predictions?

A Decision Tree is a supervised learning model that predicts by splitting data using decision rules.
It starts at the root, follows feature-based conditions, and outputs a class (or value) at a leaf node.

2. What does it mean for a node to be pure or impure?

Pure node: all samples belong to the same class

Impure node: samples belong to multiple classes

Trees aim to make nodes as pure as possible.

3. What is the role of Entropy and Gini impurity in trees?

They measure impurity in a node.
Lower values mean more homogeneous nodes.
They help decide where to split the data.

4. What is Information Gain, and why is it used?

Information Gain measures how much impurity decreases after a split.
Trees choose splits that maximize Information Gain to improve class separation.

5. Why are Decision Trees considered greedy algorithms?

Because at each node they choose the best local split (highest gain)
without considering future splits or global optimality.

6. Why do deep trees tend to overfit?

Deep trees:

Learn noise and small patterns

Create many narrow rules
This fits training data too closely and reduces generalization.

7. Explain the bias–variance tradeoff for shallow vs deep trees.

Shallow trees → high bias, low variance (underfitting)

Deep trees → low bias, high variance (overfitting)

8. What is pruning, and why is it necessary?

Pruning removes unnecessary branches from a tree.
It reduces overfitting, improves generalization, and simplifies the model.

9. Why do Decision Trees not require feature scaling?

Because trees use threshold comparisons, not distance-based calculations.
Feature magnitude does not affect split decisions.

10. Give one advantage and one limitation of Decision Trees.

Advantage: Easy to interpret and visualize

Limitation: Prone to overfitting if not controlled"""