
weight_distribution <- function(mu, S) {
  list(mu = mu, S = S)
}
  
prior_distribution <- function(a) {
  mu = c(0, 0)
  S = diag(1/a, nrow = 2, ncol = 2)
  weight_distribution(mu, S)
}

sample_weights <- function(n, distribution) {
  MASS::mvrnorm(n = n, mu = distribution$mu, Sigma = distribution$S)
}

prior <- prior_distribution(1)

plot_lines <- function(w) {
  for (i in 1:nrow(w)) {
    abline(a = w[i, 1], b = w[i, 2])
  }
}


plot(c(-1, 1), c(-1, 1), type = 'n',
     xlab = '', ylab = '')
(w <- sample_weights(5, prior))
plot_lines(w)

fit_posterior <- function(x, y, b, prior) {
  mu0 = prior$mu
  S0 = prior$S
  
  X = matrix(c(rep(1, length(x)), x), ncol = 2)
  
  S = solve(S0 + b * t(X) %*% X)
  mu = S %*% (solve(S0) %*% mu0 + b * t(X) %*% y)
  
  weight_distribution(mu = mu, S = S)
}

x <- rnorm(100)
y <- 0.2 + 1.3 * x + rnorm(100)
plot(x, y)

posterior <- fit_posterior(x, y, 1, prior)
w <- sample_weights(5, posterior)
plot_lines(w)
