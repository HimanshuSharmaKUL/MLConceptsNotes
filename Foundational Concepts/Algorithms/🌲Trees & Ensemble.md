---
sticker: emoji//1f9d9-200d-2642-fe0f
---
[[ML Algorithms]]

### Decision Tree
- Recursively split the dataset into small segments until the target variable are the same or the dataset can't be divided further

Algorithm:
1. Assign all training instances to the root of the tree. Set current node to root node.
2. Find the split-feature and split-value based on the split-criterion such as information gain, information gain ratio or gini coefficient.
3. Partition all data instances at the current node based on the split-feature and threshold value.
4. Denote each partition as a child node of the current node.
5. For each child node:
    1. If the child node is “pure” (has instances from only one class), tag it as a leaf and return.
    2. Else, set the child node as the current node and recurse to step 2.

```python
class TreeNode:
    def __init__(self, data_idx, depth, child_lst=[]):
        self.data_idx = data_idx
        self.depth = depth
        self.child = child_lst #empty list
        self.label = None
        self.split_col = None
        self.child_cate_order = None

    def set_attribute(self, split_col, child_cate_order=None):
        self.split_col = split_col
        self.child_cate_order = child_cate_order

    def set_label(self, label):
        self.label = label

class DecisionTree()
    def fit(self, X, y):
        """
        X: train data, dimensition [num_sample, num_feature]
        y: label, dimension [num_sample, ]
        """
        self.data = X
        self.labels = y
        num_sample, num_feature = X.shape
        self.feature_num = num_feature
        data_idx = list(range(num_sample))
        # Set the root of the tree
        self.root = TreeNode(data_idx=data_idx, depth=0, child_lst=[])
        queue = [self.root]
        while queue:
            node = queue.pop(0) #traverse the tree
            # Check if the terminate criterion has been met
            if node.depth>self.max_depth or len(node.data_idx)==1:
                # Set the label for the leaf node
                self.set_label(node)
            else:
                # Split the node
                child_nodes = self.split_node(node)
                if not child_nodes:
                    self.set_label(node)
                else:
                    queue.extend(child_nodes)
```



#### Entropy, Information Gain 
- Used by e ID3, C4.5 and C5.0 tree-generation algorithm
Measure of amount of uncertainty in a dataset
$$H(S) = - \displaystyle \sum_{x \in X} p(x) \times log_2(p(x))$$, where
$S$ : dataset or subset
$X$: classes to classify data into (eg.: {yes, no})
$p(x)$: proportion (probab) of elements with class $x$ over $|S|$ 
- When $H(S)=0$, the dataset is completely pure
- Expected amt of information from a distribution:
	- $E[\text{info}] = \displaystyle \sum_{x \in X} p(x) \times \text{``measure of how informative"}$
	- 1 bit → $I = -log_2(p(x))$  , 
		- Example: an observation that cuts space of possibilities in half provides one bit as $p(x) = \frac{1}{2}$, 
		  .'. $I = -log_2(\frac{1}{2})$ = 1
	- $E[\text{info}] = \displaystyle \sum_{x \in X} p(x) \times -log_2(p(x))$ : Expected amt of info/bits in $p(x)$
	- 
- Information Gain: in 1 split: measure of the difference in impurity before and after the split
$$
	{\displaystyle \overbrace {IG(S,a)} ^{\text{information gain}}=\overbrace {\mathrm {H} (S)} ^{\text{entropy (parent)}}-\overbrace {\mathrm {H} (S\mid a)} ^{\text{sum of entropies (children)}}}
$$
	-  $H(S)$: entropy of the parent node (before split)
	- $H(S \mid a)$: conditional entropy — weighted sum of the entropies of child nodes after splitting on attribute $a$
$$
H(S \mid a) = \sum_{v \in \text{Values}(a)} \frac{|S_v|}{|S|} \cdot H(S_v)
$$
			 where
			-  $\text{Values}(a)$: possible values of attribute $a$
			- $S_v$: subset of $S$ where $a = v$, i.e. subset obtained by splitting the original set S on attribute a
			- and $p(v) = \displaystyle \frac{|S_v|}{|S|}$ 
- So, Information Gain in 1 step
$$
			IG(S, a) = H(S) - \sum_{v \in \text{Values}(a)} \frac{|S_v|}{|S|} \cdot H(S_v)

$$



#### Gini Impurity, Gini Diversity Index
- Used by CART (classification and regression tree) algorithm for classification trees
- measures how often a randomly chosen element of a set would be incorrectly labeled if it were labeled randomly and independently according to the distribution of labels in the set.
Definition:
For a set of items with $J$ classes and relative frequencies $p_i$, $i \in {1, 2, ..., J}$, the probability of choosing an item with label $i$ is $p_i$, and the probability of miscategorizing that item is:

$$
\sum_{k \ne i} p_k = 1 - p_i
$$

The Gini impurity is computed by summing pairwise products of these probabilities for each class label:

$$
I_G(p) = \sum_{i=1}^{J} \left( p_i \sum_{k \ne i} p_k \right)
       = \sum_{i=1}^{J} p_i (1 - p_i)
       = \sum_{i=1}^{J} (p_i - p_i^2)
       = \sum_{i=1}^{J} p_i - \sum_{i=1}^{J} p_i^2
       = 1 - \sum_{i=1}^{J} p_i^2
$$
or, in my notes terms:
$$
  Gini(S)= 1 - \sum_{i \in X} p(x)^2
$$




#### Splitting Rules


### Bootstrapping, Bagging (Bootstrap Aggregating)
- Ensemble technique
- ↓ Variance, ↓ Overfitting
- Main idea: 
	- Introduce Diversity - by creating multiple versions of a model and then **combining them**.
	- This is done by making sure that the individual models (**classifiers**) are trained on different variations of the data
- How it works - Bootstrapping;
	1. In Bootstrapping, we randomly draw many training subsets (with replacement) from entire training data
	2. Each training data subset is used to train a different classifier of the same type
	3. Individual classifers are then combined by taking a simple majority vote of their decisions
![[Pasted image 20250723205810.png]]
- Can be Average of the decision, Majority voting etc

##### Out-of-Bag (OOB) validation
When using bagging, one can already estimate the generalization capabilities of the ensemble model using the training data: out-of-bag (OOB) validation
- 'cause an instance i will not be in every bootstrapped sample - i.e. not all classifiers would've seen that instance i, so we get some idea about generalizability w/o creating a validation set
- When validating an instance i, only consider those models which did not have i in their bootstrap sample



#### Feature Importance

#### Random Forest
The quintessential bagging technique

- Construct a multitude of decision trees at training time
- Applies bagging, so one part of randomness comes from bootstrapping each decision tree, i.e. each decision tree sees a random bootstrap of the training data - i.e. not every tree see every instance
- RF use additional piece of randomness: , i.e. to select the candidate features to split on, consider a random subset of features (sampled at every split, not once per tree!
- Individual Trees: ↑ variance, ↑overfitting, but overall, aggregated (average, majority vote) bootstrapped trees - aka Random Forest has: ↓ variance, ↓overfit 

Benefit of RF:
- Improvement upon decision trees’ habit of overftting to their training set
- No more pruning needed: each tree is overfitted on its own bootstrapped data - but by averaging we get better results, we don't need to prune to prevent overfitting

[[SciKit Learn]]
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
X, y = make_regression(n_features=4, n_informative=2,
                       random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X, y)
print(regr.predict([[0, 0, 0, 0]]))
#[-8.32987858]

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
print(clf.predict([[0, 0, 0, 0]]))
#[1]
```
Hyper parameters:
- **n_estimators**: The number of trees in the forest. To ↓overfit use ↑trees
- **max_depth**: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
- min_samples_split, min_samples_leaf
- M: size of subset of features



### Boosting
- Similar to bagging, boosting also creates an ensemble of classifiers which are then combined BUT
- Not using bootstrapping now, main idea is not diversity, but cooperation - fixing each other's mistake
- How it works-
	- classiers are added sequentially were each new classier aims at correcting the mistakes by the  ensemble thus far
![[Pasted image 20250723213009.png]]

#### AdaBoosting 
- AdaBoost trains an ensemble of weak learners (usually Logistic Reg) over a number of rounds T  
- At frst, every instance has the same weight ( $D_1 = 1/N$), so AdaBoost trains a normal classifer  
- Next, samples that were misclassified by the ensemble so far are given a heavier weight  
- The learner is also given a weight ($\alpha_t$ ) depending on its accuracy and incorporated into the ensemble  
- AdaBoost then constructs a new learner: now incorporating the weights so far
AdaBoost as additive [[ML Notes/Foundational Concepts/Algorithms/Linear, Logistic, Lasso, Ridge.md|Logistic Regression]], 


#### Gradient Boosting
What if we wanna optimise a different loss function?
- Doesn’t work with standard additive logistic regression setup or  AdaBoost  
- So take a different view: instead of weighting instances (and  classiffiers in the ensemble) in every cycle, let every sequential  classifier predict on the residuals on the ensemble so far - i.e. predict the errors instead of reweighing the instances

1. So, let true model be: $y_i = f(x_i) + \epsilon_i$ 
2. and the prediction at iteration $m$ be $F_m(x_i)$.
3. then, Gradient Boosting will fit weak learners to the residual: $y_i - F_m(x_i)$,
but we know, from [[ML Notes/Foundational Concepts/Algorithms/Linear, Logistic, Lasso, Ridge.md|Linear Regression]], the normal equations that the residuals are proportional to negative gradient of MSE loss func:

| $\frac{\partial L(\beta_0, \beta_1)}{\partial \beta_0} = -2  \sum\limits_{i=1}^{n} \left[ y_i - (\beta_0 + \beta_1 x_i) \right] = 0 \\[1em]$ ,<br>$\frac{\partial {L}(\beta_0, \beta_1)}{\partial \beta_1} = -2 \sum\limits_{i=1}^{n} \left[ y_i - (\beta_0 + \beta_1 x_i) \right] x_i = 0$ <br>that means: $\sum_{i=1}^{n} (y_i - \hat{y}_i) = 0$ or $\sum_{i=1}^{n} (e_i) = 0$ |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
saying it more genreally:
$e_{i,m} = - \displaystyle \frac{\partial {L}}{\partial F(x_i)} = y_i - F_m(x_i)$

Then Each next learner $F_{m+1}$ attempts to correct the errors of previous learner $F_m$
$$
\text{Gradient step: } F_{m+1}(x_i) = F_m(x_i) + \nu h_m(x_i)
$$
* $h_m(x_i)$: base learner fit to residuals $e_{i,m}$
* $\nu$: learning rate


#### XGBoost
- expands upon grad boost  by improving loss optimisation
- uses 2nd derivative to provide more info on direction of gradients(1st derivative: Jacobian, 2nd derivative: Hessian)



| Bagging                                                                                 | Boosting                                                                     |
| --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Can be done in parallel                                                                 | Done sequentially                                                            |
| Guarenteed ↓ variance, but not necessarily bias                                         | Guarenteed ↓ Bias, not necessarily variance                                  |
| Suitable to combine high variance low bias models (complex models) ex: RF               | Suitable to combine low variance high bias models (simple models) ex: XGB    |
| ➡ Reducing the overt of ensembles of complex models (strength of diversity/randomness) | ➡ Reducing the error of ensembles of simple models (strength of cooperation) |
![[Pasted image 20250724025932.png]]




```python
from sklearn.linear_model import Lasso, LinearRegression, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

from itertools import cycle, islice
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBClassifier, XGBRegressor
import catboost as cb
import lightgbm as lgb
```

### 1. Classification Example: Breast Cancer Dataset
```python
from sklearn.datasets import load_breast_cancer

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

```
#### Random Forest Classifier:
```python
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train)

y_pred_rf = rf_clf.predict(X_test_scaled)
print("RandomForest Classifier Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
```
#### XG Boost Classifier:
```python
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_clf.fit(X_train_scaled, y_train)

y_pred_xgb = xgb_clf.predict(X_test_scaled)
print("XGBoost Classifier Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
```

### 2. Regression Example: California Housing Dataset
```python
from sklearn.datasets import fetch_california_housing

# Load data
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

```
#### Random Forest Regressor
```python
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_scaled, y_train)

y_pred_rf = rf_reg.predict(X_test_scaled)
print("RandomForest RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False))
print("RandomForest R²:", r2_score(y_test, y_pred_rf))

```
#### XG Boost Regressor
```python
xgb_reg = XGBRegressor(n_estimators=100, random_state=42)
xgb_reg.fit(X_train_scaled, y_train)

y_pred_xgb = xgb_reg.predict(X_test_scaled)
print("XGBoost RMSE:", mean_squared_error(y_test, y_pred_xgb, squared=False))
print("XGBoost R²:", r2_score(y_test, y_pred_xgb))

```

- XGB performs better with hyperparameter tuning and early stopping
- `StandardScaler` is not strictly required for tree models but helps for consistency and when you plug in other algorithms later.