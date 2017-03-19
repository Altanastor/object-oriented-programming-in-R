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
  structure(list(mu = mu, S = S), class = "wdist")
}
  
prior_distribution <- function(a) {
  mu = c(0, 0)
  S = diag(1/a, nrow = 2, ncol = 2)
  weight_distribution(mu, S)
}
```

We give the weights distribution a class, just to distinguish them from plain lists, but otherwise there is nothing special to see here.

If we wish to sample from this distribution we can use the `mvrnorm` function from the `MASS` package.

```{r}
sample_weights <- function(n, distribution) {
  MASS::mvrnorm(n = n, 
                mu = distribution$mu,
                Sigma = distribution$S)
}

```

We can try to sample some lines from the prior distribution and plot them. We can, of course, plot the sample `w` vectors as points in the plane, but since they really represent lines, we will display them as such, see +@fig:sampling_prior.

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
S <- solve(S0 + b * t(X) %*% X)
mu <- S %*% (solve(S0) %*% mu0 + b * t(X) %*% y)
```

It is a little bit of linear algebra that involves the prior distribution and the observed values. The parameters we haven't seen before in these expressions are `b` and `X`. The former is the precision of the error terms---which we assume to be know and represent as this hyper parameter---and the latter captures the `x` values. We cannot use `x` alone because we want to use two weights to represent lines. When we write `w[1] + w[2] * x[i]` for the estimate of `y[i]`, we can think of it as the vector product of `w` and `c(0,x[i])`, which is exactly what we do. We represent all `x[i]` as rows `c(0,x[i])` in the matrix `X`. So we estimate `y[i] = w[1] * X[i, 1] + w[2] * X[I, 2]` in this notation, or `y = X %*% w`.

As a fitting function, we can write it as this:

```{r}
fit_posterior <- function(x, y, b, prior) {
  mu0 <- prior$mu
  S0 <- prior$S
  
  X <- matrix(c(rep(1, length(x)), x), ncol = 2)
  
  S <- solve(S0 + b * t(X) %*% X)
  mu <- S %*% (solve(S0) %*% mu0 + b * t(X) %*% y)
  
  weight_distribution(mu = mu, S = S)
}
```

We can try to plot some points, fit the model to these, and then plot lines sampled from the posterior. These should fall around the points now, unlike the lines sampled from the prior. The more points we use to fit the model, the tighter lines sampled from the posterior will fall around the points, see +@fig:sampling_posterior.

```{r, sampling_posterior, fig.cap = "Samples of posterior lines."}
x <- rnorm(20)
y <- 0.2 + 1.3 * x + rnorm(20)
plot(x, y)

posterior <- fit_posterior(x, y, 1, prior)
w <- sample_weights(5, posterior)
plot_lines(w)
```

```{r, echo=FALSE}
rm(x) ; rm(y)
```

The way we update the distribution of weights here from the prior distribution to the posterior doesn't require that the prior distribution is the exact one we created using the variable `a`. Any distribution can be used as the prior, and in Bayesian statistics it would not be unusual to iteratively update a posterior distribution as more and more data is added. We can start with the prior we just created, fit it to some data to get a posterior, and then if we get more data, use the first posterior as a prior for fitting more data and getting a new posterior that captures the knowledge we have gained from seeing all the data. If we have all the data from the get go, there is no particular benefit to doing this, but in an application where data comes streaming, we can exploit this to quickly update our knowledge each time new data is available.

Since this book is not about Bayesian statistics, though, and since we only use Bayesian linear regression as an example of writing a new statistical model, we will not explore this further here.


## Model matrices

The `X` matrix we used when fitting the posterior is an example of a so-called model matrix or design matrix. Using a matrix this way, to fit a response variable, `y`, to some expression, here `1 + x` (in a sense, we used the rows `c(1, x[i])`), is a general approach to fitting data. Linear models are called *linear* not because we fit data with a line, but because the weights, the `w` vector we fit, is used linearly, in the mathematical sense, in the fitted model. If we have fitted the weights to the vector `w`, the line we have fitted is given by `X %*% w`. That is, our line is the matrix product of the model matrix and the weight vector. The result is a line because `X` has the form we gave it, but it doesn't really have to represent a line for the model to be a linear model. We can transform the input data, here our vector `x`, in any way we want to, before we fit the model. You might have used log-transformations before, or fitted data to a polynomial, and those would also be examples of linear models.

You can fit various kinds of transformed data using the function we wrote above if you just construct the `X` matrix in different ways. As long as each data point you have becomes a row in `X`, it doesn't matter what you do. The same mathematics work. To fit a quadratic equation to the data instead of a line, you just have to make the i'th row of `X` be `c(1, x[i], x[i]**2)`, for example.

In R, you have a very powerful mechanism for constructing model matrices from formulas. Whenever you have used a formula such as `y ~ x + y` to fit `y` to two variables, `x` and `z`, you have used this feature. The formula is translated into a model matrix, and once you have the matrix, the code that does the model fitting doesn't need to know anything else about your data.

The function you use to translate a formula into a model matrix is called `model.matrix`. Even though it has a dot in its name, it isn't a generic function. It just has an unfortunate name for historical reasons. 

The function will take a formula as input and produce a model matrix. It will find the values for the variables in the formula by looking in its scope, and we can use it like this:

```{r}
x <- rnorm(5)
y <- 1.2 + 2 * x + rnorm(5)
model.matrix(y ~ x)
```

Relying on global variables like this is risky coding, though, so we often put our data in a data frame instead.

```{r}
d <- data.frame(x, y)
```

If we do this, we can provide a data frame to the `model.matrix` function, and it will get the variables from there.

```{r}
model.matrix(y ~ x, data = d)
```

If the formula uses variables that are not found in the data frame, the `model.matrix` will still find the missing variables in the calling scope, but this is really risky coding, so you should avoid this.

The formula determines how the model matrix is constructed. The formula `y ~ x` gives us the model matrix we used above, where we have 1 to capture the intercept in the first column, and we have the `x` values in the second column to capture the incline of the line. We can remove the intercept using the formula `y ~ x - 1`:

```{r}
model.matrix(y ~ x - 1, data = d)
```

We can also add terms, for example, we can fit a quadratic formula to the data by constructing this model matrix:

```{r}
model.matrix(y ~ x + I(x**2), data = d)
```

Here, we need to wrap the squared `x` in the function `I` to get R to actually use the squared values of `x`. Inside formulae, products are interpreted as interaction, but by wrapping `x**2` in `I` we make it the square of the `x` values explicitly.

The model matrix doesn't include the response variable, `y`, so we cannot get that from it. Instead, we can use a related function, `model.frame`, that also gives us a column for the response.

```{r}
model.frame(y ~ x + I(x**2), data = d)
```

You can then extract the response variable using the function `model.response`, like this:

```{r}
model.response(model.frame(y ~ x + I(x**2), data = d))
```

With that machinery in place, we can generalise our distributions and model fitting to work with general formulae. We can write the prior function like this:

```{r}
prior_distribution <- function(formula, a, data) {
  n <- ncol(model.matrix(formula, data = data))
  mu <- rep(0, n)
  S <- diag(1/a, nrow = n, ncol = n)
  weight_distribution(mu, S)
}
```

Ideally, a prior shouldn't depend on any data, but the form of a model matrix actually does depend on the type of the data we use. Numerical data will be represented as a single column in the model matrix, but factors are handled as a binary vector for each level in a formula, so we do need to know about what kind of data we are going to need. We could have added `n` as a parameter here, and made the function independent of any data, but I chose to include a data frame as another solution. We don't use the actual data, though, we just use it to get the number of columns in the model frame.

The function for fitting the data changes less, though. It just uses the `model.matrix` function to construct the model matrix instead of the explicit construction we did earlier:

```{r}
fit_posterior <- function(formula, b, prior, data) {
  mu0 <- prior$mu
  S0 <- prior$S
  
  X <- model.matrix(formula, data = data)
  
  S <- solve(S0 + b * t(X) %*% X)
  mu <- S %*% (solve(S0) %*% mu0 + b * t(X) %*% y)
  
  weight_distribution(mu = mu, S = S)
}
```

With that in place, we can fit a line as before, but now using a formula:

```{r}
d <- {
  x <- rnorm(5)
  y <- 1.2 + 2 * x + rnorm(5)
  data.frame(x = x, y = y)
}

prior <- prior_distribution(y ~ x, 1, d)
posterior <- fit_posterior(y ~ x, 1, prior, d)
posterior
```

If we, instead, want to fit a quadratic function, we do not need to change any of the functions, we can just provide a different formula:

```{r}
prior <- prior_distribution(y ~ x + I(x**2), 1, d)
posterior <- fit_posterior(y ~ x + I(x**2), 1, prior, d)
posterior
```

## Constructing fitted model objects

Now, we want to wrap fitted models in a class so we can write a constructor for fitting data. For fitted models, it is traditional to include the formula, the data, and the function call together with the fitted model, so we will put those as attributes in the objects. The constructor could look like this:

```{r}
blm <- function(formula, b, data, prior = NULL, a = NULL) {
  
  if (is.null(prior)) {
    if (is.null(a)) stop("Without a prior you must provide a.")
    prior <- prior_distribution(formula, a, data)
    
  } else {
    if (inherits(prior, "blm")) {
      prior <- prior$prior
    }
  }
  if (!inherits(prior, "wdist")) {
    stop("The provided prior does not have the expected type.")
  }
  
  posterior <- fit_posterior(formula, b, prior, data)
  
  structure(
    list(formula = formula,
         data = model.frame(formula, data),
         dist = posterior,
         call = match.call()),
    class = "blm"
  )
}
```

The tests at the beginning of the constructor allows us to specify the prior as either a normal distribution or a previously fitted `blm` object. If we get a prior distribution we probably should also check that this prior is compatible with the actual formula, but I will let you write such a check yourself to try that out.

If we print objects fitted using the `blm` function, they will just be printed as a list, but we can provide our own `print` function by specialising the `print` generic function. Here, it is tradition to provide the function call used to specify the model, so that is all I will do for now.

```{r}
print.blm <- function(x, ...) {
  print(x$call)
}
```

We have to match the parameters of the generic function, which is why the arguments are `x` and `...`. All we do, however, is printing the `call` attribute of the object.

Now, we can fit data and get a `blm` object like this:

```{r}
blm(y ~ x, a = 1, b = 1, data = d)
```


## Coefficients and confidence intervals



## Predicting response variables

When it comes to predicting response variables for new data, there is a little more work to be done for the model matrix. If we don't have the response variable when we create a model matrix, we will get an error, even though the model matrix doesn't actually contain a column with it. We can see this if we remove the `x` and `y` variables we used when creating the data frame earlier.

```{r}
rm(x) ; rm(y)
```

We are fine if we build a model matrix from the data frame that has both `x` and `y`:

```{r}
model.matrix(y ~ x + I(x**2), data = d)
```

If we create a data frame that only has `x` values, though, we get an error:

```{r}
dd <- data.frame(x = rnorm(5))
model.matrix(y ~ x + I(x**2), data = dd)
```

This, of course, is a problem since we are interested in predicting response variables exactly when we do not have them. Ironically, considering that the model matrix never actually contains the response variable; nevertheless, this is how R works.

To remove the response variable from a formula, before we construct a model matrix, we need to use the function `delete.response`. Unfortunately, this function does not work directly on formulas, but on so-called "terms" objects. We can translate a formula into its terms using the `terms` function. So to remove the terms from a formula, we need this construction:

```r
delete.response(terms(y ~ x))
```

The result is a `terms` object, but since the `terms` class is a specialisation of the `formula` class, we can use it directly with the `model.matrix` function to construct the model matrix:

```{r}
model.matrix(delete.response(terms(y ~ x)), data = dd)
```


