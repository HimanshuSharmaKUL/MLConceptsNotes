22-05-2025
20:00

Status:
Tags: [[Poisson]], [[Stochastic Processes]] [[Practice]] 

# Bank Stress Test

Let's break down the solution step by step and explain all the **assumptions**, **properties**, and **logic** used in solving this problem.

---

## ✅ **Problem Summary:**

- The bank's capital grows **linearly** with time:
    $C(t)=at$
    where $a>0$ is a constant rate of capital accumulation.
- Stress tests happen at **random times** according to a **Poisson process** with rate $\lambda$. This means:
    - The waiting time between stress tests is **exponentially distributed** with mean $\frac{1}{\lambda}.$
    - The stress test times are memoryless and independent.
- Each stress test demands a **random amount** of capital ss, drawn independently from a distribution with density:
$$
    f(s) = \frac{4}{\pi(1 + s^2)^2}, \quad s \ge 0
$$
    This is a **heavy-tailed distribution** similar to a squared Cauchy-like distribution, ensuring some stress tests can demand large capital.

- The bank **passes** a stress test **if**:
    $at≥s$  or  equivalently   $t \ge \frac{s}{a}$
    Otherwise, it **fails** (defaults) because it hasn't accumulated enough capital by the time of the test.

---

## 🧠 **What is being asked?**

You are asked to compute the **probability that the bank passes the stress test**, i.e., the probability that when a stress test occurs, the capital is enough to cover the required stress level.

---

## 📘 Step-by-step Explanation:

### **Step 1: Conditioning on s**

We condition on a fixed stress level $s$ and compute the probability that the bank fails **given** this stress level.

For a fixed $s$, the bank fails if the stress test occurs **before** it has time to accumulate enough capital, i.e., before $t = \frac{s}{a}.$  
So we ask:

> What is the probability that the **Poisson event** (stress test) occurs before time $\frac{s}{a}$?

From Poisson process theory, the **time of the first event** is exponentially distributed with rate $\lambda$. The probability that the first stress test occurs before time $t$ is:

$$
\mathbb{P}(T < t) = 1 - e^{-\lambda t}
$$

So for a given $s$, the **probability of failure** is:

$$
\mathbb{P}(\text{failure} \mid s) = \mathbb{P}\left(T < \frac{s}{a} \right) = 1 - e^{-\lambda s/a}
$$

---

### **Step 2: Averaging Over All Stress Levels**

Since the required stress level ss is random, we compute the **total failure probability** by averaging over all possible values of ss, weighted by their probability density f(s)f(s).

So:

$$
\mathbb{P}(\text{failure}) = \int_0^\infty f(s) \cdot \mathbb{P}\left(T < \frac{s}{a} \right) ds = \int_0^\infty f(s) \cdot \left(1 - e^{-\lambda s/a}\right) ds
$$

Simplifying:

$$
\mathbb{P}(\text{failure}) = \int_0^\infty f(s) ds - \int_0^\infty f(s) e^{-\lambda s/a} ds
$$

Since $f(s)$ is a **proper PDF**, we know:

$$
\int_0^\infty f(s) ds = 1
$$

So:

$$
\mathbb{P}(\text{failure}) = 1 - \int_0^\infty f(s) e^{-\lambda s/a} ds

$$
---

### ✅ **Final Result: Probability of Passing**

Thus, the **probability that the bank survives** (passes the test) is:

$$
\boxed{\mathbb{P}(\text{survival}) = \int_0^\infty f(s) e^{-\lambda s/a} ds}
$$

$$
$$
Plugging in the specific form of $f(s) = \frac{4}{\pi(1 + s^2)^2}:$
$$
$$

$$
\mathbb{P}(\text{survival}) = \frac{4}{\pi} \int_0^\infty \frac{e^{-\lambda s/a}}{(1 + s^2)^2} ds

$$
---

## 📌 Key Assumptions & Properties Used:

1. **Linear growth** of capital: $C(t) = at$
    
2. **Poisson process** for stress tests:
    
    - First event occurs at exponentially distributed time $t \sim \text{Exp}(\lambda)$
        
    - Memoryless property allows simple analysis using $\mathbb{P}(T < s/a)$
        
3. **Stress requirement ss** is independent and follows a specific continuous PDF.
    
4. **Conditioning** on stress level ss, then using **law of total probability** (integrating over ss).
    
5. **Exponential integral trick**: Using $\mathbb{P}(T < t) = 1 - e^{-\lambda t}$
    

---

## 🔍 Extra Note

While the original solution writes:

$$
\frac{4}{\pi} \int_0^\infty \frac{e^{-\lambda s/a}}{(1 + s^2)^2} ds
$$

there is no closed-form expression for this integral in elementary functions. It represents the **Laplace transform** of the function $\frac{4}{\pi(1 + s^2)^2}$, evaluated at $\lambda /$ a.

---

Let me know if you’d like a numerical evaluation of the survival probability for specific aa and λ, or a plot of how it behaves!
# References