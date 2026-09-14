# Machine Learning

**Machine Learning**: to extract information and identify patterns.

*Why do we think this works?* We assume it exists:

$$Y = f(x_1, \dots, x_p) + \varepsilon$$

where $\varepsilon$ is **noise**, we ask it to be $E(\varepsilon) = 0$ and $Var(\varepsilon) = \sigma^2$.

Also called **Statistical Learning**.

For which tasks?
- **Prediction**
- **Inference**: we wish to know $f$, what relation there is between $X$ and $Y$ (an *explainable* model)

### Variables
- $X$ / input variables / covariates / predictors / features
- $Y$ / output / response / target / label

---

## Tasks

### Prediction

You have $X$, but not $Y$. Being $\langle \varepsilon \rangle = 0$, then we can say that:

$$\hat{Y} = \hat{f}(X)$$

- $\hat{Y}$ → prediction for $Y$
- $\hat{f}$ → estimate for $f$

When predicting $Y$ knowing $X_1, \dots, X_p$ → a **black box model**.

**How accurate is $\hat{Y}$ to $Y$?** It depends on:
- **Reducible error** — depends on the model $\hat{f}$; in principle we could reduce this by improving the model
- **Irreducible error** — depends on $\varepsilon$, which we cannot measure

Derivation:

$$E(Y-\hat{Y})^2 = E[f(X)+\varepsilon-\hat{f}(X)]^2 = E\left[\big(f(X)-\hat{f}(X)\big)^2\right] + E(\varepsilon^2) + 2E\left[\varepsilon\big(f(X)-\hat{f}(X)\big)\right]$$

$$= E\left[\big(f(X)-\hat{f}(X)\big)^2\right] + Var(\varepsilon)$$

(using $E(\varepsilon^2) = Var(\varepsilon)$ and $E(\varepsilon)=0$)

- $E[(f(X)-\hat f(X))^2]$ → **Reducible error**
- $Var(\varepsilon)$ → **Irreducible error**

### Inference

Rather than getting a good accuracy on $\hat{Y}$, we would like to understand the relation between $X$ and $Y$.

Answering questions such as:
1. Which predictors are important?
2. Which predictors have a positive/negative relation with $Y$?
3. Is the relation "simple" (e.g., linear) or not?

---

## Estimation of $\hat{f}$

Say we have a number of points $n$ — we call these points **training data**: predictors $x_1, \dots, x_n$ and labels $y_1, \dots, y_n$.

We want to find $\hat{f}$ such that $Y \approx \hat{f}(X)$ for every $(X,Y)_{i=1}^{n}$.

We do this with:

### Parametric methods
1. Assume a shape for $f$. The simplest shape? $f$ is linear.
2. This model, whichever it is, will have $(p+1)$ parameters, $\beta_0, \dots, \beta_p$.
3. Then the problem is to find the $(p+1)$ parameters. How?

The fact that we jump from estimating $f$ to estimating $\beta_0 \dots \beta_p$ results in a distance between $\hat{f}$ and $f$. To reduce this distance, we choose **flexible models**, that means more parameters. The downside of this is that we can adapt the model too much to the set of data we have.

*(For a linear model, the simplest approach is the least squares.)*

### Non-parametric methods
There is no assumption on the shape of $f$, but they try to estimate $f$ directly with the data points.
- **Con**: they need a lot of data points.
- There is also the problem of **overfitting**.

### Why prefer one approach to another?
It depends on the question I'm trying to answer.

- **Inference** → a simple, few-parameter model is preferable to a model full of parameters, or with complicated shapes, and highlights the importance of single predictors.
- **Prediction** → a good algorithm, very accurate, is preferable to a simpler method, even though the algorithm becomes a "black box."

---

## Supervised and Unsupervised Learning

- **Supervised**: we have a response $y_i$ for each $x_i$ → here we can look for the relation between $X$ and $Y$, and also perform prediction.
- **Unsupervised**: we have the predictors, but no response $y_i$ → here we can look for the relation between different $X_i, X_j$ (e.g., cluster analysis).
- Also **semi-supervised learning**.

## Regression and Classification Problems

- Quantitative response → **regression**
- Qualitative response → **classification**

The focus is on the response. The nature of the predictors is less important.

---

## How good is a model?

For example, how far is the predicted response from the actual response? When working in **regression**, we have the **Mean Squared Error (MSE)**:

$$MSE = \frac{1}{n}\sum_{i=1}^{n}\big(y_i - \hat{f}(x_i)\big)^2$$

- $n$ → number of data points
- $y_i$ → response
- $\hat{f}(x_i)$ → prediction

One wants MSE **small**.

**Which data points?** Those of a subset of the labeled data called the **training dataset** ⇒ **training MSE**.

We need to use the training dataset to produce a model that fits well **not** the training dataset, but the **future data points**.

### Flexibility vs. MSE (U-shape)

- **Linear regression**: not very flexible, low degrees of freedom, underfitting.
- **Smoothing spline**: very flexible, high degrees of freedom, overfitting.

**Fundamental property**: for every dataset, for each statistical method, the test MSE follows a **U-shape** as a function of flexibility.

- The smoothing spline with the highest number of parameters has the lowest **training MSE** (it fits very well the data points $(x_i,y_i)$, $i=1,\dots,n$) but performs badly on new points $(x_0,y_0)$ with $0 \notin \{1,\dots,n\}$ — it has the highest **test MSE**.

We need the model for which:

$$Avg\big(y_0-\hat{f}(x_0)\big)^2 $$

is smallest.