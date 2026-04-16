---
title: "DS4420 Final Project Bayesian Model"
output: html_document
date: "2026-04-14"
---

````{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
````

````{r}

# Load libraries
library(mvnfast)

# Helper function
rinvchisq_custom <- function(n, df, scale) {
  (df * scale) / rchisq(n, df = df)
}

# Load data
data <- read.csv("/Users/rhoda/Downloads/parkinsons_updrs.data.csv")


# Train/test split
set.seed(42)
n_total <- nrow(data)
indices <- sample(1:n_total)
train_size <- floor(0.8 * n_total)
train_idx <- indices[1:train_size]
test_idx  <- indices[(train_size + 1):n_total]

train_data <- data[train_idx, ]
test_data  <- data[test_idx, ]

# Target variable
y_train <- train_data$motor_UPDRS
y_test  <- test_data$motor_UPDRS


# Drop columns we do not want
X_train_raw <- train_data[, !(names(train_data) %in% c("motor_UPDRS", "total_UPDRS", "subject#", "index"))]
X_test_raw  <- test_data[,  !(names(test_data)  %in% c("motor_UPDRS", "total_UPDRS", "subject#", "index"))]


# Add nonlinear features
X_train_sq <- X_train_raw^2
colnames(X_train_sq) <- paste0(colnames(X_train_raw), "_sq")

X_test_sq <- X_test_raw^2
colnames(X_test_sq) <- paste0(colnames(X_test_raw), "_sq")

# Add interactions
X_train_int <- data.frame(
  HNR_PPE      = X_train_raw$HNR * X_train_raw$PPE,
  HNR_RPDE     = X_train_raw$HNR * X_train_raw$RPDE,
  age_testtime = X_train_raw$age * X_train_raw$test_time,
  PPE_DFA      = X_train_raw$PPE * X_train_raw$DFA
)

X_test_int <- data.frame(
  HNR_PPE      = X_test_raw$HNR * X_test_raw$PPE,
  HNR_RPDE     = X_test_raw$HNR * X_test_raw$RPDE,
  age_testtime = X_test_raw$age * X_test_raw$test_time,
  PPE_DFA      = X_test_raw$PPE * X_test_raw$DFA
)

X_train_raw <- cbind(X_train_raw, X_train_sq, X_train_int)
X_test_raw  <- cbind(X_test_raw,  X_test_sq,  X_test_int)


# Standardize predictors
train_means <- apply(X_train_raw, 2, mean)
train_sds   <- apply(X_train_raw, 2, sd)
train_sds[train_sds == 0] <- 1

X_train_scaled <- scale(X_train_raw, center = train_means, scale = train_sds)
X_test_scaled  <- scale(X_test_raw,  center = train_means, scale = train_sds)

# Add intercept
X_train <- cbind(1, X_train_scaled)
X_test  <- cbind(1, X_test_scaled)


# OLS estimate (regularized)
lambda <- 1.0
XtX   <- t(X_train) %*% X_train
I_mat <- diag(ncol(X_train))
I_mat[1, 1] <- 0

w_hat <- solve(XtX + lambda * I_mat) %*% t(X_train) %*% y_train
print("OLS weights:")
print(w_hat[1:5, ])


# Posterior sampling
n <- nrow(X_train)
p <- ncol(X_train)

resid     <- y_train - X_train %*% w_hat
scale_val <- as.numeric(t(resid) %*% resid / (n - p))

sigma2_samples <- rinvchisq_custom(
  n     = 5000,
  df    = n - p,
  scale = scale_val
)

w_samples <- matrix(0, nrow = 5000, ncol = p)
for (i in 1:5000) {
  w_samples[i, ] <- rmvn(
    1,
    mu    = as.vector(w_hat),
    sigma = solve(XtX + lambda * I_mat) * sigma2_samples[i]
  )
}


# Posterior plots
hist(w_samples[, 2], main = "Posterior of Feature 1 Coefficient", xlab = "Value")
hist(w_samples[, 3], main = "Posterior of Feature 2 Coefficient", xlab = "Value")
hist(w_samples[, 4], main = "Posterior of Feature 3 Coefficient", xlab = "Value")


# Posterior predictive distribution
pred_samples <- c()
for (i in 1:5000) {
  pred_samples <- c(
    pred_samples,
    rnorm(
      1,
      mean = sum(w_samples[i, ] * X_train[1, ]),
      sd   = sqrt(sigma2_samples[i])
    )
  )
}

hist(pred_samples, main = "Posterior Predictive Distribution", xlab = "Predicted motor_UPDRS")

cat("Posterior predictive mean:", mean(pred_samples), "\n")
cat("Actual motor_UPDRS:", y_train[1], "\n")


# Test predictions using posterior mean
w_mean     <- colMeans(w_samples)
y_pred_test <- X_test %*% w_mean

mse_test  <- mean((y_test - y_pred_test)^2)
rmse_test <- sqrt(mse_test)

ss_res_test <- sum((y_test - y_pred_test)^2)
ss_tot_test <- sum((y_test - mean(y_test))^2)
r2_test     <- 1 - ss_res_test / ss_tot_test

cat("Test MSE:",  mse_test,  "\n")
cat("Test RMSE:", rmse_test, "\n")
cat("Test R^2:",  r2_test,   "\n")

# Predicted vs Actual plot
plot(
  y_test, y_pred_test,
  main = "Bayesian: Predicted vs Actual ",
  xlab = "Actual motor_UPDRS",
  ylab = "Predicted motor_UPDRS",
  pch  = 16,
  col  = rgb(0, 0, 1, 0.35)
)
abline(0, 1, col = "red", lty = 2, lwd = 2)



# brms Gaussian Model 
library(brms)


# Prepare data for brms
# Use the original (non-squared) features from train_data,
# scaled the same way as above, keeping it as a data frame
feature_cols <- names(train_data)[!(names(train_data) %in%
                                      c("motor_UPDRS", "total_UPDRS", "subject#", "index"))]

parkinsons_scaled <- data[, c(feature_cols, "motor_UPDRS")]

# Scale the predictors using the full dataset means/sds
feat_means <- apply(parkinsons_scaled[, feature_cols], 2, mean)
feat_sds   <- apply(parkinsons_scaled[, feature_cols], 2, sd)
feat_sds[feat_sds == 0] <- 1

parkinsons_scaled[, feature_cols] <- scale(
  parkinsons_scaled[, feature_cols],
  center = feat_means,
  scale  = feat_sds
)


# Check default priors
prior_gaussian <- c(
  prior(normal(0, 10), class = b),
  prior(normal(20, 10), class = Intercept),
  prior(exponential(1), class = sigma)
)


# Fit brms Gaussian model
park_brm <- brm(
  motor_UPDRS ~ age + sex + test_time + Jitter... + Shimmer +
                NHR + HNR + RPDE + DFA + PPE +
                HNR:PPE + HNR:RPDE + age:test_time + PPE:DFA +
                I(HNR^2) + I(PPE^2) + I(DFA^2) + I(RPDE^2),
  family  = gaussian(),
  data    = parkinsons_scaled,
  chains  = 4,
  cores   = getOption("mc.cores", 1),
  iter    = 1000,
  warmup  = 180,
  thin    = 2,
  prior   = prior_gaussian
)

# Summary & convergence plots
summary(park_brm)
plot(park_brm)


# Bayesian R^2
bayes_R2(park_brm)


# Posterior predictive distribution for first observation
post_preds <- posterior_predict(park_brm)

# First patient in the dataset
patient1_post_preds <- post_preds[, 1]
hist(patient1_post_preds,
     main = "Posterior Predictive Distribution (Patient 1)",
     xlab = "Predicted motor_UPDRS")

cat("Posterior predictive mean (Patient 1):", mean(patient1_post_preds), "\n")
cat("Actual motor_UPDRS (Patient 1):", parkinsons_scaled$motor_UPDRS[1], "\n")



# Test set R^2 for brms (to comapire with manual model)
test_df <- parkinsons_scaled[test_idx, ]

y_pred_brms <- predict(park_brm, newdata = test_df)[, "Estimate"]
y_test_brms <- test_df$motor_UPDRS

ss_res_brms <- sum((y_test_brms - y_pred_brms)^2)
ss_tot_brms <- sum((y_test_brms - mean(y_test_brms))^2)
r2_brms_test <- 1 - ss_res_brms / ss_tot_brms

rmse_brms_test <- sqrt(mean((y_test_brms - y_pred_brms)^2))

cat("brms Test R^2:", r2_brms_test, "\n")
cat("brms Test RMSE:", rmse_brms_test, "\n")


# Conditional effects plots
plot(conditional_effects(park_brm, "age"))
plot(conditional_effects(park_brm, "HNR"))
plot(conditional_effects(park_brm, "PPE")) 

````
