---
title: Decision Tree
---

## How splitting works

The more we split, the better the accuracy is, but not every time. We have to choose the features to split upon.

- We check which feature gives us maximum information gain, and based on that we split our data into.
- Say we have features x1, x2, x3, x4.
- If 1st split is on x1, then we split the rest on the remain x2,x3,x4, which ever gives the maximum gain. Say on the 2nd we get max gain on x2 split, 3rd on x3 and then x4.
- If the node is pure, then just output the class.
- And if the node is still not pure and there is no feature left to split, then we output majority class.
- If both the above condition is not met, then -
    - Find the best feature to split upon and
    - Recursively call upon the split.

## How to find the best feature

Objective metrics

- Accuracy
- Information gain
- Gain ration
- Gini index

### Accuracy
We check which feature gives the best accuracy, then we split upon that feature.

Suppose there are few features like college tier, internship, project done or not. And our goal is to find if he/she will get a job or not.

- Say 1st we split about tier of college, we get accuracy of 0.84.
- Then we check by splitting about internship and say we get accuracy of 0.91.
- And we get accuracy of 0.8 based on project feature.
- So, we see internship feature gave the best split. Thus, we choose the internship feature to make the 1st split.
- Then for the 2nd split, we consider the project and college tier feature to find out the best accuracy split and then split accordingly.

### Information Gain

Suppose we have two class Yes "y" and NO "n".

- Let there be 9y and 20n.
- We have entropy or information required.
- The more pure the node is, the lower the entropy is and the less information it is required to get rid of the impurities.
- In case of pure node, entropy is 0.

We will pick the feature to split upon which will give us the max information gain.

### Gain Ratio

It is the ration of info gain and split info $\frac{information-gain}{split-information}$

It will try to split into unbalanced nodes, where few of them will be pure and few will require further splitting.

### Gini Index

- It is $Gini Index = 1 - \sum{pi^2}$

where, pi is the probability of the classes.

## How to overcome overfitting?
- Limit the max depth
- Limit to min change in metric.
- Pruning








