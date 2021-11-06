# Decision Tree
It is an academic project of DecisionTree
INDE577 (Rice University) 
Thanks to Professor Randy R. Davila.

## Decision Tree
* Supervised learning
- classification
- regression (Regression Trees)

* Each node has an "imparit measure" of your choice
    - A pure node contains only samples of the sample class;
    - We want all leaves to be pure , and pure node to have " an okay amount of data points".

## The Formula of Inpurity of a node
- Entropy = \sum_{i=1}^c (- P_i) * log_2(P_i) 
- Gini = 1 - \sum_{i=1}^c (P_i)^2
Here c is the total numver of classes and P_i is the probability of class i in the node.

## Entropy
Basically, it is the measurement of the impurity or randomness in the data points.
“Entropy is a degree of randomness or uncertainty, in turn, satisfies the target of Data Scientists and ML models to reduce uncertainty.”


## Gini Index
Gini Index, also known as Gini impurity, calculates the amount of probability of a specific feature that is classified incorrectly when selected randomly. If all the elements are linked with a single class then it can be called pure.

How to calculate Gini, for example:
There are 11 points with 5 red points and 6 blue points in the root node V. 
- Gini of the root node V is : Gini(root node) = 1- (5/11)^2-(6/11)^2
Then divide V into V_L and V_R, there are 3 red in V_L, 2 red points + 11 blue point in V_R,
- Gini of V_L is : Gini(V_L)=1-(3/3)^2 - (0/3)^2=0
- Gini of V_R is: Gini(V_R)=1-(2/13)^2-(11/13)^2

## Information Gain (IG)
Information Gain is applied to quantify which feature provides maximal information about the classification based on the notion of entropy, i.e. by quantifying the size of uncertainty, disorder or impurity, in general, with the intention of decreasing the amount of entropy initiating from the top (root node) to bottom(leaves nodes).

We start at node V choose a condition to split into node V_L and node V_R
- IG = I(V) - (W_L * I(V_L) + W_R * I(V_R))
- W_L = I(V_L)/I(V);
- W_R = I(V_R)/I(V).
What we try to do is to find the split that maximize this IG.


In addition, decision tree algorithms exploit Information Gain to divide a node and Gini Index or Entropy is the passageway to weigh the Information Gain.

## Gini Index vs Information Gain
Take a look below for the getting discrepancy between Gini Index and Information Gain,
- The Gini Index facilitates the bigger distributions so easy to implement whereas the Information Gain favors lesser distributions having small count with multiple specific values.
- The method of the Gini Index is used by CART algorithms, in contrast to it, Information Gain is used in ID3, C4.5 algorithms.
- Gini index operates on the categorical target variables in terms of “success” or “failure” and performs only binary split, in opposite to that Information Gain computes the difference between entropy before and after the split and indicates the impurity in classes of elements.
