# Decision Trees, Decision Boundaries, and Evaluation
# A decision tree is a type of non-parametric supervised learning that can be used for both regression and classification.
# In this notebook we explore classification decision trees on an artificial dataset.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# The artificial data will be tatken from sklearn (make_moons)
from sklearn.datasets import make_moons

#X,y = make_moons(noise=0.01, random_state = 3)
# change noise from 0.01 to 04
X,y = make_moons(noise=0.4, random_state = 3) 
colors = ["red" if label == 0 else "blue" for label in y]

plt.figure(figsize = (10,8))
plt.scatter(X[:,0],X[:,1],c=colors)
plt.xlabel("feature x_0")
plt.ylabel("feature x_1")
plt.show()



from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,
                                                  y,
                                                 test_size = 0.4,
                                                 random_state = 42)
c_train = ["red" if label == 0 else "blue" for label in y_train]
c_test = ["red" if label == 0 else "blue" for label in y_test]

plt.figure(figsize = (10,8))
plt.scatter(X_train[:,0],X_train[:,1],c=c_train)
plt.xlabel("feature x_0")
plt.ylabel("feature x_1")
plt.show()


features = ["x_0","x_1"]
labels = ["red","blue"]

"""
----
We will use the Decision Tree Claasifier from sklearn. 
Documentation can be found at tree 
(https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

----
"""

from sklearn.tree import DecisionTreeClassifier

decision_tree=DecisionTreeClassifier(max_depth=2,random_state=42)
decision_tree.fit(X_train,y_train)

# plot the decision_tree
from sklearn import tree

plt.figure(figsize=(20,8))
a = tree.plot_tree(decision_tree,
                  feature_names = features,
                  class_names = labels,
                  rounded = True,
                  filled = True,
                  fontsize = 14)
plt.show()


#show the result of tree in text way
from sklearn.tree import export_text
tree_rules = export_text(decision_tree,
                        feature_names = features)
print(tree_rules)
#Result
#|--- x_0 <= -0.30
#|   |--- x_1 <= 0.10
#|   |   |--- class: 0
#|   |--- x_1 >  0.10
#|   |   |--- class: 0
#|--- x_0 >  -0.30
#|   |--- x_1 <= 0.36
#|   |   |--- class: 1
#|   |--- x_1 >  0.36
#|   |   |--- class: 0

# In the following code cell we show the decision boundaries from our trained tree


# Set a plot_step
plot_step = 0.02

plt.figure(figsize=(10,8))
# Plot the decision boundary
x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
y_min,y_max = X[:,1].min()-1,X[:,1].max()+1

xx,yy= np.meshgrid(np.arange(x_min,x_max,plot_step),
                   np.arange(y_min,y_max,plot_step))

#plt.tight_layout(h_pad=0.5,w_pad=0.5,pad=2.5)

Z = decision_tree.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
cs= plt.contourf(xx,yy,Z,cmap='jet_r')


plt.scatter(X_train[:,0],X_train[:,1],c=c_train)
plt.xlabel("feature x_0")
plt.ylabel("feature x_1")
plt.show()

"""
----
## The Confusion Matrix

In many instances we are interested in the following:
 * true positieves -> predicted true and actually true
 * false positives -> predicted true and not actually true
 * false negatives -> predicted false but actually true
 * true negnatives -> predicted false and actually false
 
All of these possibilities are contained in the confusion matrix.

----
"""

# predicted values on the testing data
test_pred_decision_tree = decision_tree.predict(X_test)

# Import metrics from sklearn
from sklearn import metrics

# Note: visualizing your tree above will be weird after running seaborn
import seaborn as sns

# The confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test,test_pred_decision_tree)

# Convert confusion matrix into dataframe
matrix_df = pd.DataFrame(confusion_matrix)

plt.figure(figsize=(10,8))
ax = plt.axes()
sns.set(font_scale=1.3)

sns.heatmap(matrix_df,
           annot=True,
           fmt="g",
           ax=ax,
           cmap="magma")

ax.set_title("Confusion Matrix - Decision Tree")
ax.set_xlabel(labels)
ax.set_ylabel("True Label",fontsize=10)
ax.set_yticklabels(labels,rotation=0)
plt.show()

# 5 blue dots in red
# 2 red dots in blue
# 20 red dots in red
# 13 blue dots in blue

## plt.imshow(confusion_matrix)

# Set a plot_step with the test data, which is related the confusion_matrix above
## then we can see the meaning of result from confusion_matrix above more easily
plot_step = 0.02
plt.figure(figsize=(10,8))
# Plot the decision boundary
x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
y_min,y_max = X[:,1].min()-1,X[:,1].max()+1
xx,yy= np.meshgrid(np.arange(x_min,x_max,plot_step),
                   np.arange(y_min,y_max,plot_step))
#plt.tight_layout(h_pad=0.5,w_pad=0.5,pad=2.5)
Z = decision_tree.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
cs= plt.contourf(xx,yy,Z,cmap='jet_r')
plt.scatter(X_test[:,0],X_test[:,1],c=c_test)
plt.xlabel("feature x_0")
plt.ylabel("feature x_1")
plt.show()

# show the accuracy of the prediction
print(f"accuracy score={metrics.accuracy_score(y_test,test_pred_decision_tree)}")
# Result : accuracy score=0.825


# Precision tell us how many of the values we predicted to be in a certain class are actually in that class!
print(f"precision score")
precision = metrics.precision_score(y_test,
                                   test_pred_decision_tree,
                                   average=None)
precision_results = pd.DataFrame(precision,index=labels)
precision_results.rename(columns = {0:"precision"},inplace = True)
precision_results

# Recall and the f1-score (look them up!)
print(metrics.classification_report(y_test,test_pred_decision_tree))
#precision score
#              precision    recall  f1-score   support
#           0       0.80      0.91      0.85        22
#           1       0.87      0.72      0.79        18
#    accuracy                           0.82        40
#   macro avg       0.83      0.82      0.82        40
#weighted avg       0.83      0.82      0.82        40
