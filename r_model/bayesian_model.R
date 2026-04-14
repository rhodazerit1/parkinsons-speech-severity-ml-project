
---
title: "DS4420 Final Project Baeysian Model"
output: html_document
date: "2026-04-14"
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Description

A Bayesian linear regression model was implemented to estimate the relationship between speech features and Parkinson’s disease severity. 

Posterior distributions for model parameters were obtained using sampling methods, allowing uncertainty quantification in both coefficients and predictions.

Model performance was evaluated using standard regression metrics and compared to other approaches.

```{r Baeysian Model}

# -----------------------------
# Load libraries
# -----------------------------
library(mvnfast)

# -----------------------------
# Helper function
# -----------------------------
rinvchisq_custom <- function(n, df, scale) {
  (df * scale) / rchisq(n, df = df)
}

# -----------------------------
# Load data
# -----------------------------
data <- read.csv("/Users/rhoda/Downloads/parkinsons_updrs.data.csv")

# Target variable
y <- data$motor_UPDRS

# Drop columns we do not want
X_raw <- data[, !(names(data) %in% c("motor_UPDRS", "total_UPDRS", "subject#", "index"))]

# Standardize predictors
X_scaled <- scale(X_raw)

# Add intercept
X <- cbind(1, X_scaled)

# -----------------------------
# OLS estimate
# -----------------------------
w_hat <- solve(t(X) %*% X) %*% t(X) %*% y

print("OLS weights:")
print(w_hat[1:5, ])

# -----------------------------
# Posterior sampling
# -----------------------------
n <- nrow(X)
p <- ncol(X)

resid <- y - X %*% w_hat
scale_val <- as.numeric(t(resid) %*% resid / (n - p))

sigma2_samples <- rinvchisq_custom(
  n = 5000,
  df = n - p,
  scale = scale_val
)

w_samples <- matrix(0, nrow = 5000, ncol = p)

for (i in 1:5000) {
  w_samples[i, ] <- rmvn(
    1,
    mu = as.vector(w_hat),
    sigma = solve(t(X) %*% X) * sigma2_samples[i]
  )
}

# -----------------------------
# Posterior plots
# -----------------------------
hist(w_samples[, 2], main = "Posterior of Feature 1 Coefficient", xlab = "Value")
hist(w_samples[, 3], main = "Posterior of Feature 2 Coefficient", xlab = "Value")
hist(w_samples[, 4], main = "Posterior of Feature 3 Coefficient", xlab = "Value")

# -----------------------------
# Posterior predictive distribution
# -----------------------------
pred_samples <- c()

for (i in 1:5000) {
  pred_samples <- c(
    pred_samples,
    rnorm(
      1,
      mean = sum(w_samples[i, ] * X[1, ]),
      sd = sqrt(sigma2_samples[i])
    )
  )
}

hist(pred_samples, main = "Posterior Predictive Distribution", xlab = "Predicted motor_UPDRS")
cat("Posterior predictive mean:", mean(pred_samples), "\n")
cat("Actual motor_UPDRS:", y[1], "\n")

# -----------------------------
# Simple fitted values using posterior mean
# -----------------------------
w_mean <- colMeans(w_samples)
y_pred <- X %*% w_mean

mse <- mean((y - y_pred)^2)
rmse <- sqrt(mse)

ss_res <- sum((y - y_pred)^2)
ss_tot <- sum((y - mean(y))^2)
r2 <- 1 - ss_res / ss_tot

cat("MSE:", mse, "\n")
cat("RMSE:", rmse, "\n")
cat("R^2:", r2, "\n")

```
