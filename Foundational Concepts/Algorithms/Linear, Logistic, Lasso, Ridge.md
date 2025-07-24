---
sticker: emoji//1f34e
---
[[ML Algorithms]] , [[Bias vs Variance Trade Off]], [[Multicolinarity]]

### Linear Regression

![[Pasted image 20250723125848.png|300]]

In linear regression, the observations (**red**) are assumed to be the result of random deviations (**green**) from an underlying relationship (**blue**) between a dependent variable (_y_) and an independent variable (_x_).


[[SciKit Learn]] implementation:
$$
\hat{y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p 
$$
My Notes:
$$
y_i = \beta_o + \beta_1x_{i1} + \beta_2x_{i2} + ... + \beta_{p-1}x_{i,p-1} + \epsilon_i
$$

#### Assumptions
(Linearity, Homoscedasticity, Normality, No multicollinearity)
Assumptions for the error term $\epsilon_i$  (Gauss-Markov Conditions)
0. Errors are Normally distributed with zero mean
1. Expected value: $E[\epsilon_i]=0$
2. Variance of error terms $Var[\epsilon_i]=\sigma^2$ : Homoscedasticity i.e. constant variance i.e. variance of the errors does not depend on the values of the predictor variables. Thus the variability of the responses for given fixed values of the predictors is the same regardless of how large or small the responses are.
3. 0 Covariance of error terms: $\mathrm{Cov}[\epsilon_i, \epsilon_j] = \mathbb{E}[\epsilon_i \epsilon_j] - \mathbb{E}[\epsilon_i] \mathbb{E}[\epsilon_j] = \mathbb{E}[\epsilon_i \epsilon_j] = 0 \quad \text{for all } i \ne j$  , i.e. Errors are mutually independent
4. Error $\epsilon_i$ are independent of X 
5. **Lack of perfect multicollinearity** in the predictors. For standard [least squares](https://en.wikipedia.org/wiki/Least_squares "Least squares") estimation methods, the design matrix _X_ must have full [column rank](https://en.wikipedia.org/wiki/Column_rank "Column rank") _p_; otherwise perfect [multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity "Multicollinearity") exists in the predictor variables, meaning a linear relationship exists between two or more predictor variables.

If X is not deterministic (set by design), then X is R.V.
then, all of above assumptions are conditioned on X
.'.
Y is a r.v. that satisfies: 
$E[Y\mid X_1, X_2, ...X_{p-1}] = \beta_o + \beta_1x_{i1} + \beta_2x_{i2} + ... + \beta_{p-1}x_{i,p-1}$
i.e. We model Y as the mean, expected response



#### OLS - Ordinary Least Sq Estimator
Aim: to choose such a set of coefficients ( $\beta_0, \beta_1,...$) that residuals are minimised
Residuals: $e_i(\hat \beta_0,\hat \beta_1) = y_i - \hat \beta_0+\hat \beta_1x_i$
Least Square Estimator to determine the params:
$(\beta_0^{LS}, \beta_1^{LS}) = \displaystyle \arg\min_{\beta_0, \beta_1} \displaystyle \sum_{i=1}^n e_i^2(\beta_0, \beta_1) = \arg\min_{\beta_0, \beta_1} \displaystyle \sum_{i=1}^n (y_i - (\beta_0 + \beta_1 x_i))^2$

i.e. mathematically, OLS solves problems of form: $\min_{w} ||y - Xw||_2^2$  (square of L2 norm)


##### [[Matrix Notation]]
- If no columns of a matrix can be expressed as a linear combination of the other, we say that the columns are linearly independent.
- rank of a matrix : 
	- is the maximum number of linearly  independent columns of a matrix  
	-  or, maximum number of linearly independent rows. 
	- Hence, the rank of an r x c matrix cannot exceed min(r,c).
- If there is correlation, Multicollinearity in matrix, then it wont be full rank
- Inverse of a Matrix: $AA^{-1} = A^{-1}A = I$
	- The inverse is only defined for square matrices.  
	- The inverse of a r x r matrix only exists if the rank is r  (nonsingular or full rank i.e. NO MULTICOLINEARITY)
$$
y = X\beta + \epsilon
$$
$$
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{bmatrix}
=
\begin{bmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1,p-1} \\
1 & x_{21} & x_{22} & \cdots & x_{2,p-1} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \cdots & x_{n,p-1}
\end{bmatrix}
\begin{bmatrix}
\beta_0 \\
\beta_1 \\
\vdots \\
\beta_{p-1}
\end{bmatrix}
+
\begin{bmatrix}
\epsilon_1 \\
\epsilon_2 \\
\vdots \\
\epsilon_n
\end{bmatrix}
$$
Assumptions of OLS:
1. Exp value of error: $\mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0}$
2. Variance of $Var(\epsilon) = \sigma^2$ ; Var-covar matrix of errors$$
\Sigma(\boldsymbol{\epsilon}) =
\begin{bmatrix}
\sigma^2 & 0 & \cdots & 0 \\
0 & \sigma^2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \sigma^2
\end{bmatrix}
= \sigma^2 \mathbf{I}_n
$$
3. Params $\hat{\boldsymbol{\beta}} = (\hat{\beta}_0, \ldots, \hat{\beta}_{p-1})^{\mathrm{T}}$ yield residuals $e_i$ and fitted values $\hat y_i$
	$e_i(\hat{\boldsymbol{\beta}}) = y_i - \hat{y}_i = y_i - \mathbf{x}_i^{\mathrm{T}} \hat{\boldsymbol{\beta}}$
OLS estimates $\hat{\boldsymbol{\beta}}$ by minimising SSE, Least Square Estimator objective Func :$$
\hat{\boldsymbol{\beta}}_{LS} = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} e_i^2(\boldsymbol{\beta})
$$
Differentiating $\hat{\boldsymbol{\beta}}_{LS} = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} e_i^2(\boldsymbol{\beta})$, wrt each $\beta_j (j=0,1,..p-1)$ and setting derivatives = 0 gives normal eqns. 
$$
\frac{\partial}{\partial \boldsymbol{\beta}} \left( \|\mathbf{y} - \mathbf{X} \boldsymbol{\beta}\|^2 \right) = -2\mathbf{X}^{\mathrm{T}} (\mathbf{y} - \mathbf{X} \boldsymbol{\beta}) = 0
$$

| $\frac{\partial L(\beta_0, \beta_1)}{\partial \beta_0} = -2  \sum\limits_{i=1}^{n} \left[ y_i - (\beta_0 + \beta_1 x_i) \right] = 0 \\[1em]$ ,<br>$\frac{\partial {L}(\beta_0, \beta_1)}{\partial \beta_1} = -2 \sum\limits_{i=1}^{n} \left[ y_i - (\beta_0 + \beta_1 x_i) \right] x_i = 0$ <br>that means: $\sum_{i=1}^{n} (y_i - \hat{y}_i) = 0$ or $\sum_{i=1}^{n} (e_i) = 0$ |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

$\mathbf{X}^{\mathrm{T}} \mathbf{X} \hat{\boldsymbol{\beta}} = \mathbf{X}^{\mathrm{T}} \mathbf{y}$ (aka Normal Eqn of Linear Reg)
Now: If rank(X) = p $\le$ n (p: number of coeff, n: number of datapoints), then sol. of $\mathbf{X}^{\mathrm{T}} \mathbf{X} \hat{\boldsymbol{\beta}} = \mathbf{X}^{\mathrm{T}} \mathbf{y}$ is given by:
$\hat{\boldsymbol{\beta}} = (\mathbf{X}^{\mathrm{T}} \mathbf{X})^{-1}\mathbf{X}^{\mathrm{T}} \mathbf{y}$ 

now comes the concept of multicollinearity
#### Multicollinearity
and Variance Inflation Factor (VIF)
- The condition rank(X) = p $\le$ n  is 'cause $\mathbf{X}^{\mathrm{T}} \mathbf{X}$ is a $p \times p$ matrix, .'. only invertible when it is full rank i.e. rank is p
- and, the rank of $\mathbf{X}^{\mathrm{T}} \mathbf{X}$ can't be p when rank of $\mathbf{X}$ is not p, e.g. when $n<p$
- If rank($\mathbf{X}$) = p, the solution to $\hat{\boldsymbol{\beta}} = (\mathbf{X}^{\mathrm{T}} \mathbf{X})^{-1}\mathbf{X}^{\mathrm{T}} \mathbf{y}$  is unique, analytically deriviable
	- ![[Pasted image 20250723143444.png|200]]
-  If rank($\mathbf{X}$) < p, the solution to $\hat{\boldsymbol{\beta}} = (\mathbf{X}^{\mathrm{T}} \mathbf{X})^{-1}\mathbf{X}^{\mathrm{T}} \mathbf{y}$  is non unique, infinite number of LS solutions
	- ![[Pasted image 20250723143522.png|250]]
- More realistic: the $\mathbf{X}$-variables might be strongly correlated  (multicollinearity):
	- ![[Pasted image 20250723143712.png|300]]
	- In such a case, the LS fit is uniquely defined, but many other  parameter estimates $\hat \beta$ attain a residual sum of squares which is close to the minimal value of $\hat \beta_{LS}$
	- Small changes in the data set may cause a large change in the parameter estimates

![[Pasted image 20250723144250.png|700]]

Centering can reduce multicollinearity

#### Variance of Errors $\sigma^2$
$\sigma^2$ can be estimated by Mean Square Error (MSE)
( Mean Square Error = Variance + Bias$^2$)
$$
\hat \sigma^2 = s^2 = \displaystyle \frac{1}{n-p}\sum_{i=1}^{n} e_i^2
$$

ANOVA Decomposition Equation:
$$
\sum_{i=1}^{n} (y_i - \bar{y})^2 = \sum_{i=1}^{n} (\hat{y}_i - \bar{y})^2 + \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \\
$$
$$
SST = SSR + SSE
$$
$$
Sum Sq Total = Sum Sq Reg + Sum Sq Error 
$$
$SSR = \beta'X'Y' - \displaystyle \frac{1}{n}Y'JY$, $SSE = Y'Y - \beta'X'Y$
 $MSR= \displaystyle \frac{SSR}{p-1}$, $MSE = \displaystyle \frac{SSE}{n-p}$ ,

$R^2$:Coefficient of Multiple Determination
$$
R^2 = \frac{\text{SSR}}{\text{SST}} = 1 - \frac{\text{SSE}}{\text{SST}} 
= 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$


 
We can decompose MSE into Bias and Variance

$$
MSE = \mathbb{E}_{D, \varepsilon} \left[ \left( y - \hat{f}(x; D) \right)^2 \right] 
= \left( \operatorname{Bias}_D\left[ \hat{f}(x; D) \right] \right)^2 
+ \operatorname{Var}_D\left[ \hat{f}(x; D) \right] 
+ \sigma^2
$$
where, $\displaystyle D=\{(x_{1},y_{1})\dots ,(x_{n},y_{n})$ is training data, 
and for Linear Reg, $\hat{f}(x) = x^\top \hat{\beta}$ , 
Bias is defined as: 
$$
\operatorname{Bias}_D \left[ \hat{f}(x; D) \right] 
= \mathbb{E}_D \left[ \hat{f}(x; D) \right] - f(x)
= \mathbb{E}_D \left[ \hat{f}(x; D) \right] - \mathbb{E}_{y \mid x}[y]
$$
- Bias is basically: $Bias = E[\hat f(x)] - f(x)$, i.e. error bw avg model prediction and ground truth
Var is:
$$
\operatorname{Var}_D \left[ \hat{f}(x; D) \right] 
= \mathbb{E}_D \left[ \left( \hat{f}(x; D) - \mathbb{E}_D[\hat{f}(x; D)] \right)^2 \right]
$$
- Variance of the predicted model is nothing but direct variance formula: $Var[\hat f(x)] =  E[(\hat f(x) - E[\hat f(x)])^2]$, 
Irreducible error term:
$$
\sigma^2 = \mathbb{E}_{y} \left[ \left( y - f(x) \right)^2 \right] 
= \operatorname{Var}(\varepsilon)
$$



### Bias vs Variance Trade Off

#### Over vs Under Fitting
Explanatory variable reduction because:  
- too many predictors hard to maintain, costly, less interpretable; irrelevant vars ↑ prediction variance
- multicollinearity (highly correlated vars) → unstable coeffs, ↑ variance, roundoff errors
- On the other hand, omitting important variables (or latent  explanatory variables) leads to biased estimates of
	- the regression coefficients, $\beta_j (j=0,1,..p-1)$
	- the error variance $\sigma$, ('.' $Var(\epsilon) = \sigma^2$ ), 
	- the mean responses and predictions of new observations.

![[Pasted image 20250723172320.png|400]]

High Bias$^2$:
- Underfitting, oversimple model, both high train (... line) and test ( _ solid line) error
High Variance:
- Overfitting, overcomplex model
- Low Train (... line) error, High test ( _ solid) error


What is Bias Variance Trade off:
Dilemma to simultaneously reduce the 2 sources of error: Bias and Variance
-  ↑ Bias → ↓ Variance and ↓ Bias → ↑ Variance
- 
What is Double Descend phenomenon
![[Pasted image 20250723172722.png]]



### Regularisation
Put penalty on size of the weights:
- ↑ generalisation, ↓ overfitting, ↓prone to outliers
- "Penalty" means the optimisation function has to account for this extra term now

#### Lasso Regression (L1 Regularisation):
Minimises L1 norm: $argmin_\beta \displaystyle \underbrace{\sum_{i=1}^{n}(y_i - \hat y_i^2)}_{\text{SSE, Residual/Error}} + \lambda \displaystyle \sum_{j=1}^{p}|\beta_j|$,    $p$ is no. of params
$|\beta_j|$ is not differentiable


#### Ridge Regression
Minimises L2 norm: $argmin_\beta \displaystyle \underbrace{\sum_{i=1}^{n}(y_i - \hat y_i^2)}_{\text{SSE, Residual/Error}} + \lambda \displaystyle \sum_{j=1}^{p}(\beta_j)^2$

[`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge "sklearn.linear_model.Ridge") regression addresses some of the problems of [Ordinary Least Squares](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares) by imposing a penalty on the size of the coefficients. The ridge coefficients minimize a penalized residual sum of squares: $$\min_{w} || X w - y||_2^2 + \alpha ||w||_2^2$$
#### Properties of Lasso and Ridge
- For Lasso and Ridge, standardisation and normalisation is requried

|     | Lasso                                                                                                                               | Ridge                                                                                                                                                                  | Notes |
| --- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
|     | force coefcients to become zero                                                                                                     | only keeps them within bounds                                                                                                                                          |       |
|     | Variable selection “for free” with Lasso![[Pasted image 20250723183554.png]] i.e. variable selection via coefficient shrinkage to 0 | Easier to implement (slightly) and faster to compute (slightly), or when you have a limited number of variables to  begin with<br>![[Pasted image 20250723183650.png]] |       |
|     | In practice lasso tends to work well even with small sample sizes                                                                   |                                                                                                                                                                        |       |
|     | Normalise variable beforehand                                                                                                       | Normalise variables beforehand so that normalisation term $\lambda$ affects the variables in similar manner                                                            |       |
|     |                                                                                                                                     |                                                                                                                                                                        |       |


#### Elasstic Net: Combo of Lasso, Ridge
Elastic net: $argmin_\beta \displaystyle \underbrace{\sum_{i=1}^{n}(y_i - \hat y_i^2)}_{\text{SSE, Residual/Error}} + \lambda_1 \displaystyle \sum_{j=1}^{p}|\beta_j| + \lambda_2 \displaystyle \sum_{j=1}^{p}(\beta_j)^2$,


### Logistic Regression

