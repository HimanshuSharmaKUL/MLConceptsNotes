---
aliases: []
---
### Cross-validation: evaluating estimator performance
- best params found by [[Grid Search]] techniques
- then do a random test-train split
```python
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

X_train.shape, y_train.shape
X_test.shape, y_test.shape
#SVM
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)
```
##### Need for CV:
Improper workflow (leaky):    
    - Train a model with some hyperparameters.
    - Evaluate it on the **test set**.
    - Change hyperparameters.
    - Try again on the test set.
    - Repeat...
Even if unintentionally, this process **uses the test set to guide decisions**, which leads to overfitting to the test set.
✅ 1. Proper workflow (no leak):
- **Split** your data:
    - `training set`: used to fit the model.
    - `validation set`: used to tune hyperparameters.
    - `test set`: used _only once_, at the very end.
- **Tuning process**:
    - Try different hyperparameters (like `C` in SVM) using cross-validation on the training set (or validation set).
    - Pick the best model based on validation performance.
    - Finally, evaluate the selected model on the test set once

Two challanges:
- by partitioning the available data into three sets, we drastically reduce the number of samples which can be used for learning the model, and 
- the results can depend on a particular random choice for the pair of (train, validation) sets.
Solution: Cross Validation -> A test set should still be held out for final evaluation, but the **validation set is no longer needed when doing CV**

#### K-Fold Cross Val
- training set is split into _k_ smaller sets
- The following procedure is followed for each of the _k_ “folds”:
	- A model is trained using  $k-1$ of the folds as training data;
	- the resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).
- performance measure reported by _k_-fold cross-validation is then the average of the values computed in the loop
![[Pasted image 20250722173131.png]]

Example: estimate the accuracy of a linear kernel support vector machine on the iris dataset by 
1. splitting the data, 
2. fitting a model and 
3. computing the score 5 consecutive times (with different splits each time):
```python
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
scores
array([0.96, 1. , 0.96, 0.96, 1. ])
```
mean score and the standard deviation are hence given by
```python
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
```

---
The following sections list utilities to generate indices that can be used to generate dataset splits according to different cross validation strategies.
WE USE FOLLOWING FOR MORE CONTROL
Otherwise, just use cross_val_score()
#### KFold

```python
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in kf.split(X): #<============== KFold generates indices for train and test 
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
```
![[Pasted image 20250722174845.png]]

#### Stratified KFold
We can see that [`StratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold "sklearn.model_selection.StratifiedKFold") preserves the class ratios (approximately 1 / 10) in both train and test datasets.
![[Pasted image 20250722175146.png]]

#### Leave One Group Out
 - cross-validation scheme where each split holds out samples belonging to one specific group. Group information is provided via an array that encodes the group of each sample.
 - Each training set is thus constituted by all the samples except the ones related to a specific group. This is the same as [`LeavePGroupsOut`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePGroupsOut.html#sklearn.model_selection.LeavePGroupsOut "sklearn.model_selection.LeavePGroupsOut") with `n_groups=1` and the same as [`GroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html#sklearn.model_selection.GroupKFold "sklearn.model_selection.GroupKFold") with `n_splits` equal to the number of unique labels passed to the `groups` parameter.

