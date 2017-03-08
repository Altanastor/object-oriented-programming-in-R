# Statistical models

Truth be told, you won't be using object-oriented programming in most day to day R programming. Most analyses you do in R involve transformation of data, typically implemented as some sort of data flow, that is much better captured by functional programming. When you write such pipelines, you will probably be using polymorphic functions, but you rarely need to create your own classes. If you need to implement a new statistical model, however, you usually do want to create a new class.

A lot of data analysis requires that you infer parameters of interest or you build a model to predict properties of your data, but in many of those cases you don't necessarily need to know exactly how the model is constructed, how it infers parameters, or how it predicts new values. You can use the `coefficents` function to get inferred parameters from a fitted model or you can use the `predict` function to predict new values and you can use those two functions with almost any statistical model. That is because most models are implemented as classes with implementations for the generic functions `coefficients` and `predict`.

As an example of object-oriented programming in action, we can implement our own model in this chapter. We will keep it simple, so we can focus on the programming aspects and not the statistical theory, but still implement something that isn't already built into R. We will implement a version of Bayesian linear regression.

## Bayesian linear regression

The simplest form of linear regression is fitting a line to data points. Imagine we have vectors `x` and `y`, we wish to produce coefficients `w[1]` and `w[2]` such that `y[i] = w[1] + w[2] x[i] + e[i]` where the `e` is a vector of errors that we want to make as small as possible (we typically assume that the errors are identically normally distributed when we consider it a statistical problem and so we want to have the minimal variance for the errors). When fitting linear models with the `lm` function, you getting the maximum likelihood values for the weights `w[1]` and `w[2]`, but if you wish to do Bayesian statistics you should instead consider this weight vector `w` as a random variable, and fitting it to the data in `x` and `y` means updating it from its prior distribution to its posterior distribution.

A typical distribution for linear regression weights is the normal distribution. If we consider the weights multivariate normal distributed as their prior distribution then their posterior distribution given the data will also be normally distributed, which makes the mathematics very convenient.

We will assume that the prior distribution of `w` is a normal distribution with mean zero and independent components, so a diagonal covariance matrix. This means that, on average, we believe the line we are fitting to be flat and going through the plane origin, but how strongly we believe this depend on values in the covariance matrix. This we will parameterise with a so-called *hyper parameter*, `a`, that is the precision---one over the variance---of the weight components. The covariance matrix will have `1/a` on its diagonal and zeros off-diagonal.

We can represent a distribution over weights as the mean and covariance matrix of a multinomial normal distribution and construct the prior distribution from the precision like this:

```{r}
weight_distribution <- function(mu, S) {
  list(mu = mu, S = S)
}
  
prior_distribution <- function(a) {
  mu = c(0, 0)
  S = diag(1/a, nrow = 2, ncol = 2)
  weight_distribution(mu, S)
}
```

If we wish to sample from this distribution we can use the `mvrnorm` function from the `MASS` package.

```{r}
sample_weights <- function(n, distribution) {
  MASS::mvrnorm(n = n, 
                mu = distribution$mu,
                Sigma = distribution$S)
}

```

We can try to sample some lines from the prior distribution and plot them. We can, of course, plot the sample `w` vectors as points in the plane, but since they really represent lines, we will display them as such.

```{r, sampling_prior, fig.cap="Samples from the prior of lines."}
prior <- prior_distribution(1)
(w <- sample_weights(5, prior))

plot(c(-1, 1), c(-1, 1), type = 'n',
     xlab = '', ylab = '')
plot_lines <- function(w) {
  for (i in 1:nrow(w)) {
    abline(a = w[i, 1], b = w[i, 2])
  }
}
plot_lines(w)
```

When we observe data in the form of matching `x` and `y` values, we must updated the `w` vector to reflect this, which means updating the distribution of the weights. I won't derive the math, this is not a math textbook after all, but if `mu0` is the prior mean and `S0` the prior covariance matrix, then the posterior mean and covariance matrix are computed thus:

```r
S = solve(S0 + b * t(X) %*% X)
mu = S %*% (solve(S0) %*% mu0 + b * t(X) %*% y)
```

It is a little bit of linear algebra that involves the prior distribution and the observed values. The parameters we haven't seen before in these expressions are `b` and `X`. The former is the precision of the error terms---which we assume to be know and represent as this hyper parameter---and the latter captures the `x` values. We cannot use `x` alone because we want to use two weights to represent lines. When we write `w[1] + w[2] * x[i]` for the estimate of `y[i]`, we can think of it as the vector product of `w` and `c(0,x[i])`, which is exactly what we do. We represent all `x[i]` as rows `c(0,x[i])` in the matrix `X`. So we estimate `y[i] = w[1] * X[i, 1] + w[2] * X[I, 2]` in this notation, or `y = X %*% w`.

As a fitting function, we can write it as this:

```{r}
fit_posterior <- function(x, y, b, prior) {
  mu0 = prior$mu
  S0 = prior$S
  
  X = matrix(c(rep(1, length(x)), x), ncol = 2)
  
  S = solve(S0 + b * t(X) %*% X)
  mu = S %*% (solve(S0) %*% mu0 + b * t(X) %*% y)
  
  weight_distribution(mu = mu, S = S)
}
```

We can try to plot some points, fit the model to these, and then plot lines sampled from the posterior. These should fall around the points now, unlike the lines sampled from the prior. The more points we use to fit the model, the tighter lines sampled from the posterior will fall around the points.

```{r, sampling_posterior, fig.cap = "Samples of posterior lines."}
x <- rnorm(20)
y <- 0.2 + 1.3 * x + rnorm(20)
plot(x, y)

posterior <- fit_posterior(x, y, 1, prior)
w <- sample_weights(5, posterior)
plot_lines(w)
```

## Model matrices

