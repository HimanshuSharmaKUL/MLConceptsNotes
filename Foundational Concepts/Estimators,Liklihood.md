24-05-2025
21:28

Status:
Tags:[[Liklihood]], 


# Estimators

**fundamental conceptual distinction** in statistics between the **likelihood function** and the **probability model** (or simply, the **probability function**).

Let’s break this down clearly:

---

### 🔁 Likelihood vs. Probability (or PMF/PDF)

#### 1. **Probability Function (PMF or PDF)**

- **Parameters are fixed** (e.g., p=0.6p = 0.6),
- **Data is variable** (e.g., “What’s the probability of observing X=2X = 2?”).
- You're asking: **What’s the probability of the data, assuming a known model?**

$$
P(X = x \mid p)
$$

🟩 This is the standard use of probability in forward modeling: given a fixed $p,$ what is the probability of seeing data like $x$?

---
Now we flip our perspective. 
#### 2. **Likelihood Function**

- **Data is fixed** (e.g., you observed x = 2),
- **Parameter is variable** (e.g., try different values of p).
- You're asking: **How plausible are different values of the parameter given the data?**

$$
L(p \mid x) = P(X = x \mid p)
$$

But now you're treating x as fixed and exploring p as variable.

🟨 This is used in estimation — especially **Maximum Likelihood Estimation (MLE)**.

---
### 🔄 General Steps to Write a Likelihood Function

1. **Specify the distribution of your data.**
    - Identify the probability mass function (pmf) or probability density function (pdf) of your random variable, depending on whether it's discrete or continuous.

2. **Write the joint probability of the observed data.**
    - First write the joint probab distribution. If they're iid, then good, just take product of indiv pdf/pmfs , otherwise, take the joint distribution of the observed data. Then flip the pov from fixed param, variable data (in probab calulation case) to fixed data, variable param (in parameter estimation/inference case)
    - If you have a sample $x_1, x_2, \dots, x_n$ assumed i.i.d. (independent and identically distributed), then the ==likelihood is the product of individual pmfs/pdfs evaluated at each data point==

3. **Express the likelihood as a function of the parameter(s).**
    - Treat the data as fixed and the parameter(s) as variable.

4. **(Optional) Take the log to get the log-likelihood.**
    - This simplifies optimization problems and turns products into sums.

---

### ✅ Your Case: $X \sim \text{Binomial}(3, p)$

Let’s suppose you observe one value $X = x$, or more generally, a sample of observations $x_1, x_2, \dots, x_n$ from this distribution.

#### Step 1: PMF of the Binomial(3, p)

$$
P(X = x) = \binom{3}{x} p^x (1 - p)^{3 - x}, \quad x = 0, 1, 2, 3
$$

#### Step 2: Joint Likelihood for a Sample

If you observe data $x_1, x_2, \dots, x_n,$ then the **likelihood function** is:

$$
L(p) = \prod_{i=1}^{n} \binom{3}{x_i} p^{x_i} (1 - p)^{3 - x_i}
$$

#### Step 3: Simplify (if needed)

You can simplify the likelihood:

$$
L(p) = \left[ \prod_{i=1}^n \binom{3}{x_i} \right] \cdot p^{\sum x_i} (1 - p)^{3n - \sum x_i}
$$

This is now a function of p, with data fixed.

#### Step 4: Log-Likelihood (optional, often used for MLE)

$$
\log L(p) = \sum_{i=1}^n \log \binom{3}{x_i} + \left(\sum x_i\right) \log p + \left(3n - \sum x_i\right) \log(1 - p)
$$

The term $\sum \log \binom{3}{x_i}$ doesn’t depend on $p$, so it can be ignored for maximization.

---

### Summary

For any distribution:

- Use its pmf/pdf
- Multiply over the sample (if i.i.d.)
- Express as a function of the unknown parameter(s)
- Optionally take the log for simplicity




### ❓ So What Happens If You Treat Parameter as Fixed and Data as Variable?

✅ Then you are back to the **probability function**.

This is **not** called a likelihood function anymore — it’s just the **probability mass function (pmf)** for discrete variables or **probability density function (pdf)** for continuous ones.

---

### 🧭 Summary Table

|Function Type|Parameters|Data|Name|Used For|
|---|---|---|---|---|
|Probability Model|Fixed|Variable|PMF/PDF|Simulating or computing probs|
|Likelihood Function|Variable|Fixed|Likelihood|Inference, parameter estimation|

---

### Can we calculate the MLE without knowing the sample values?
No, you **cannot** calculate the Maximum Likelihood Estimate (MLE) of a parameter (like $p$) **without knowing the observed data values**.

### 🔄 Reminder: What Is MLE?

The **MLE** of a parameter (e.g. $\hat{p}$) is the value that **maximizes the likelihood function**:

$$
\hat{p} = \arg\max_{p} \, L(p \mid \text{data})
$$

The likelihood function depends **entirely on the data** — it's a function of the unknown parameter **given the data**. So without data, you can't write the full likelihood function, and therefore you can't optimize it.

---

### ⚠️ What If We Know the Distribution, But Not the Data?

If you're told:

- $X_i \sim \text{Binomial}(3, p)$
- You have **5 observations**
- But you’re **not given the values** $x_1, x_2, \dots, x_5$
Then all you can say is that:

$$
L(p) = \prod_{i=1}^5 \binom{3}{x_i} p^{x_i} (1 - p)^{3 - x_i}
$$

But this expression contains unknown values $x_i$, so it is **incomplete**.

You **cannot simplify** or maximize this expression to get $\hat{p}$ without:

- the values of $x_1, \dots, x_5$, or
- at least the **sufficient statistic** — in this case, $\sum x_i$

---

### ✅ However, If You Know the $\sum x_i$...

In the binomial case, the **sufficient statistic** is:
$$
T = \sum_{i=1}^n x_i
$$

So if you **don't know the individual values**, but you **do know T**, you _can_ write the likelihood in terms of T:

$$
L(p) \propto p^{T}(1 - p)^{3n - T}
$$

This is enough to compute the MLE:

$$
\hat{p} = \frac{T}{3n}
$$

✅ This result makes sense because the binomial distribution is part of the exponential family, and sufficient statistics contain all the information needed for estimation.

---

### 🔚 Conclusion

|Do you have...|Can you compute MLE?|
|---|---|
|No data at all|❌ No|
|The full dataset|✅ Yes|
|Only the sum ∑xi\sum x_i|✅ Yes (in binomial)|



# References