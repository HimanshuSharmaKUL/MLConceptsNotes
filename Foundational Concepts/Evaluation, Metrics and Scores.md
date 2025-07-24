---
sticker: emoji//1f920
---

### Regression Metrics

#### $R^2$ - Coefficient of Determination

- proportion of variance (of y) that has been explained by the independent variables in the model\
-  indication of goodness of fit and therefore a measure of how well unseen samples are likely to be predicted by the model, through the proportion of explained variance.
- As such variance is dataset dependent,  may not be meaningfully comparable across different datasets.
- Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected (average) value of y, disregarding the input features, would get an  score of 0.0.
$$
R^2(y, \hat{y}) = 1 - \displaystyle \frac{\displaystyle \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }{\displaystyle \sum_{i=1}^{n} (y_i - \bar{y})^2}
$$
$\displaystyle \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ : Residual Sum Squares
$\sum_{i=1}^{n} (y_i - \bar{y})^2$: Total Sum Squares

#### Adjusted $R^2$ 



### Imbalanced Dataset
[[Imbalanced Data]] 
#### Data is Imbalanced

#### Misclassification Costs
- We're willing to make some mistakes on the negative side to catch the positives
- 



