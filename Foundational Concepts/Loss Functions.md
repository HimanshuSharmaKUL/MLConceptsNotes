15-07-2025
17:59

Status:
Tags: [[Liklihood]]

# Loss Functions

## Cross Entropy Loss

**Definition**: Cross-entropy measures the dissimilarity between two probability distributions — typically, the predicted distribution $\hat{y}​$ (e.g., from a softmax layer) and the true distribution $y$ (usually a one-hot encoded label).
##### Formula:
For a single example:

$$
\text{CE}(y, \hat{y}) = -\sum_{i=1}^C y_i \log(\hat{y}_i)
$$
- C = number of classes
- $y_i$= 1 if the class is the correct one, else 0
- $\hat{y}_i$ = predicted probability for class i
Since $y$is one-hot, only one term remains:

$$
\text{CE}(y, \hat{y}) = -\log(\hat{y}_{\text{true class}})
$$

**Use case**: Standard for multi-class classification with softmax outputs.
## Negative Log Likelihood (NLL)

**Definition**: NLL is the negative of the log-probability assigned to the correct class. It's equivalent to cross-entropy loss when the predicted values are already log-probabilities (i.e., after a `log_softmax` layer).

##### Formula:
$$
\text{NLL}(y, \log \hat{y}) = -\log(\hat{y}_{\text{true class}})
$$
**Use case**: Often used **in combination with `log_softmax`**, especially in PyTorch:

```python
nn.NLLLoss()  # requires log-probabilities`
```

##### Q. If we have C classes, what is the expected NLL loss?
If you have C classes and **the model predicts uniformly at random**, **expected Negative Log-Likelihood (NLL) loss** is:

$$
E[\text{NLL}] = -\log\left(\frac{1}{C}\right) = \log(C)
$$
'prediciting uniformly at random' means model has no clue and just guesses randomly i.e. $\log{C}$ is worst case prediction
If predictions are uniform, then for any true class $y$, the model assigns:

$$
\hat{y}_i = \frac{1}{C} \quad \text{for all } i = 1, 2, ..., C
$$
So for the correct class (say class $j$), the negative log-likelihood is:
$$
\text{NLL} = -\log\left(\frac{1}{C}\right) = \log(C)
$$

This holds for every class under uniform prediction, so the **expected loss over all classes** is also $\log(C)$ 
- A good model will have NLL < $\log(C)$.
- This baseline is useful for judging performance — for example:
    - For 10-class classification, $\log(10) \approx 2.302$
    - For 1000 classes, $\log(1000) \approx 6.91$

