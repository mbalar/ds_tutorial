---
layout: post
title: Machine Learning - Python for Data Science
---

*This is the 4th and final post in a series meant to be an accelerated learning guide, allowing you to quickly setup your environment and start playing around with data using Python (importing data, working with dataframes, referencing, filtering, cleaning, feature engineering, visualizing, machine learning, etc.)*

### Splitting Data

The first thing we'll do is split our data into a training set and a testing set. The training set will be used to "train" our model and the testing set will be used to ensure that we haven't overfitted that model. A good model generalizes well and can be applied to unseen data while providing similar performance, hence the seperate testing set to validate this notion.

A widely used machine learning library, scikit-learn, includes functionality to randomly split our dataframe:


```python
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split

train, test = train_test_split(df, test_size = 0.20)
```

This created two new dataframes with 80% of the data in train and 20% in test, which we'll now convert to arrays using the Pandas values method:


```python
train_data = train.values
test_data = test.values

test_data
```




    array([[ 0.        ,  1.        ,  0.        , ...,  0.        ,
             0.        ,  0.03125   ],
           [ 0.        ,  1.        ,  0.        , ...,  0.        ,
             0.        ,  0.02678571],
           [ 1.        ,  1.        ,  1.        , ...,  0.        ,
             0.        ,  0.02564103],
           ..., 
           [ 1.        ,  1.        ,  1.        , ...,  0.        ,
             0.        ,  0.04878049],
           [ 1.        ,  1.        ,  1.        , ...,  0.        ,
             0.        ,  0.05284553],
           [ 1.        ,  1.        ,  1.        , ...,  0.        ,
             0.        ,  0.03448276]])



Definitely not as pretty nor digestible as the dataframe representation, but this array conversion is necessary before fitting our model.

### Random Forest

A random forest is an ensemble of decision trees developed for classification, which is particulary useful for our purpose as we're trying to determine a 0/1 response (software developer vs. non-developer). I highly recommend taking a look at this [Visual Introduction to Machine Learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/ "Visual Intro") before moving on to see exactly how a decision tree and random forest works.

We'll use scikit-learn's RandomForestClassifier to create a forest containing 100 trees and pass our training data through:


```python
from sklearn.ensemble import RandomForestClassifier 

x_train = train_data[0::,1::]  # Features
y_train = train_data[0::,0]    # IsSoftwareDev

 # Random forest object with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

 # Fit the training data to the IsSoftwareDev labels and create the decision trees
forest = forest.fit(x_train,y_train)
```

### Model Evaluation

Using the score method, we can determine the accuracy of the random forest predictions:


```python
forest.score(x_train,y_train)
```




    1.0



The training accuracy is 100%, i.e. all of the observations were classified as software developers (or not) correctly. This shouldn't be shocking since our model was fit on the training data. Let's see how it does on the unseen test set:


```python
x_test = test_data[0::,1::]
y_test = test_data[0::,0]

forest.score(x_test,y_test)
```




    0.7359550561797753



The test accuracy is only 74%. The mismatches are due to overfitting, but by tweaking the model, assumptions, and features we may be able to improve the accuracy.

Here's the receiver operating characteristic (ROC) curve:


```python
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

y_pred_rf = forest.predict_proba(x_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

plt.style.use('ggplot')
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()
```


![png](https://mbalar.github.io/img/output_19_0.png)


The area under the curve is a measure of model accuracy and we can use the breakouts below to evaluate our model:

.90-1.0 = Excellent (A)  
.80-.90 = Good (B)  
.70-.80 = Fair (C)  
.60-.70 = Poor (D)  
.50-.60 = Fail (F)  

To calculate the area under curve (AUC):


```python
from sklearn import metrics

metrics.auc(fpr_rf, tpr_rf)
```




    0.83131236579512446



Overall, not bad for our first attempt but there's still some room for improvement.

### Feature Importance

Finally, let's take a look at which features prodived the greatest predictive value. The feature_importances method will calculate these values and we'll use matplotlib to create a bar plot to visualize them:


```python
%matplotlib inline
import matplotlib.pyplot as plt

feature_names = df.columns.values[1:]
importances = forest.feature_importances_

imp = pd.DataFrame(importances, index=feature_names).sort_values(0)

imp.plot.barh(figsize=(8,6), fontsize=12)
```








![png](https://mbalar.github.io/img/output_25_1.png)


Without getting into what the numbers actually mean, the feature we created from MonthsProgramming and AgeFill ended up having the highest relative importance. Finishing the bootcamp program, using multiple learning resources, going to different types of code events, and listening to coding podcasts all ranked highly as well.

### Wrap up

Ideally, we'd also do an out-of-time validation. If FreeCodeCamp conducts the same survey again next year we could really test how well our random forest model is at predicting software developers.

A lot of material has been covered over the last few posts and I would highly encourage going back and seeing if there's anything that could have been done differently to yield better results, e.g. try a different machine learning technique, select a different set of features, feature transformations, etc.

### Additional Reading

[Pandas Documentation](http://pandas.pydata.org/pandas-docs/stable/index.html "Pandas Documentation")  
[Matplotlib Documentation](http://matplotlib.org/contents.html "Matplotlib Documentation")  
[Scikit-learn Documentation](http://scikit-learn.org/dev/index.html "Scikit Documentation")
