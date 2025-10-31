########################################################################
# apple_forecasting.R
# Apple (AAPL) monthly stock forecasting (2019-01 to 2024-12)
# Models: ETS (Holt-Winters) and ARIMA (auto.arima)
# Metrics: RMSE, MAE, MAPE
# Author: Hamza Humayun
########################################################################

# -----------------------------
# 0. Install / load required packages
# -----------------------------
required_pkgs <- c("quantmod", "forecast", "tseries", "ggplot2", "dplyr", "readr")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[,"Package"])]
if(length(new_pkgs)) install.packages(new_pkgs)

library(quantmod)   # for getSymbols
library(forecast)   # for ets, auto.arima, forecast, accuracy
library(tseries)    # for adf.test (if needed)
library(ggplot2)    # for nicer plots
library(dplyr)      # data manipulation
library(readr)      # write_csv

# -----------------------------
# 1. Data download and preprocessing
# -----------------------------
# Download AAPL daily data from Yahoo (2019-01-01 to 2024-12-31)
getSymbols("AAPL", src = "yahoo", from = "2019-01-01", to = "2024-12-31", auto.assign = TRUE)

# Convert to monthly and extract adjusted/close price
# to.monthly returns OHLC; we'll take adjusted close if present (AAPL$AAPL.Adjusted)
# quantmod stores the object AAPL
apple_monthly <- to.monthly(AAPL, indexAt = "lastof", name = "AAPL", OHLC = TRUE)

# Choose Adjusted if available, else Close
if ("AAPL.Adjusted" %in% colnames(AAPL)) {
  apple_close_monthly <- monthlyReturn(AAPL[, "AAPL.Adjusted"], type = "log") # not used; keep original below
  # simpler: extract last adjusted value of each month:
  monthly_prices <- period.apply(AAPL[, "AAPL.Adjusted"], endpoints(AAPL, on = "months"), last)
} else {
  monthly_prices <- period.apply(AAPL[, "AAPL.Close"], endpoints(AAPL, on = "months"), last)
}

# Convert to numeric vector & time series
monthly_index <- index(monthly_prices)
monthly_values <- as.numeric(coredata(monthly_prices))

# Create ts object starting Jan 2019 with frequency 12
apple_ts <- ts(monthly_values, start = c(2019, 1), frequency = 12)

# Quick plot of original series
plot(apple_ts, main = "Apple Monthly Prices (2019-01 to 2024-12)", ylab = "Price (USD)", xlab = "Year")

# Save cleaned data to CSV for reproducibility
if(!dir.exists("data")) dir.create("data")
df_prices <- data.frame(Date = as.Date(monthly_index), Close = monthly_values)
write_csv(df_prices, "data/apple_monthly_prices_2019_2024.csv")

# -----------------------------
# 2. Split data for evaluation (training/test)
# -----------------------------
# We'll use the last 12 months (2024) as test (hold-out) to evaluate forecasting accuracy.
h <- 12  # forecast horizon in months

n <- length(apple_ts)
train_ts <- window(apple_ts, end = c(2019 + (n-h-1) %/% 12, (n-h) %% 12 + 1))
# simpler approach: use time-based subsetting:
train_end <- time(apple_ts)[n - h]
train_ts <- window(apple_ts, end = train_end)
test_ts <- window(apple_ts, start = time(apple_ts)[n - h + 1])

# Print lengths to confirm
cat("Total months:", n, "\n")
cat("Training months:", length(train_ts), "\n")
cat("Test months (holdout):", length(test_ts), "\n")

# -----------------------------
# 3. Model 1: ETS (Holt-Winters via ets)
# -----------------------------
# ets() will choose the best exponential smoothing model automatically
ets_model <- ets(train_ts)
summary(ets_model)

# Forecast h months
ets_forecast <- forecast(ets_model, h = h, level = c(80, 95))

# -----------------------------
# 4. Model 2: ARIMA (auto.arima)
# -----------------------------
# auto.arima chooses (p,d,q)(P,D,Q)[m] automatically
arima_model <- auto.arima(train_ts, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)
summary(arima_model)

# Forecast h months
arima_forecast <- forecast(arima_model, h = h, level = c(80, 95))

# -----------------------------
# 5. Plot forecasts together
# -----------------------------
if(!dir.exists("plots")) dir.create("plots")

# ETS plot
png("plots/ets_forecast.png", width = 900, height = 600)
plot(ets_forecast, main = "ETS Forecast - Apple Monthly Prices", ylab = "Price (USD)")
lines(test_ts, col = "red", lwd = 2) # overlay actual test
legend("topleft", legend = c("Forecast", "Actual (test)"), col = c("blue", "red"), lwd = 2)
dev.off()

# ARIMA plot
png("plots/arima_forecast.png", width = 900, height = 600)
plot(arima_forecast, main = "ARIMA Forecast - Apple Monthly Prices", ylab = "Price (USD)")
lines(test_ts, col = "red", lwd = 2)
legend("topleft", legend = c("Forecast", "Actual (test)"), col = c("blue", "red"), lwd = 2)
dev.off()

# Combined ggplot (optional)
library(reshape2)
fc_df <- data.frame(
  Date = seq(as.Date("2019-01-31"), by = "month", length.out = length(apple_ts) + h),
  Actual = c(as.numeric(apple_ts), rep(NA, h))
)
# add forecasts for plotting convenience (only forecast periods)
fc_df$ETS_forecast <- c(as.numeric(rep(NA, length(apple_ts))), as.numeric(ets_forecast$mean))
fc_df$ARIMA_forecast <- c(as.numeric(rep(NA, length(apple_ts))), as.numeric(arima_forecast$mean))
fc_df$ARIMA_forecast <- c(rep(NA, length(apple_ts)), as.numeric(arima_forecast$mean))

p <- ggplot(fc_df, aes(x = Date)) +
  geom_line(aes(y = Actual), size = 0.8) +
  geom_line(aes(y = ETS_forecast), linetype = "dashed") +
  geom_line(aes(y = ARIMA_forecast), linetype = "dotted") +
  labs(title = "Apple: Actual and Forecasts (ETS dashed, ARIMA dotted)", y = "Price (USD)")
ggsave("plots/combined_forecasts.png", p, width = 10, height = 6)

# -----------------------------
# 6. Forecast evaluation: compute RMSE, MAE, MAPE on hold-out (test_ts)
# -----------------------------
# helper metrics
rmse <- function(actual, predicted) sqrt(mean((actual - predicted)^2, na.rm = TRUE))
mae  <- function(actual, predicted) mean(abs(actual - predicted), na.rm = TRUE)
mape <- function(actual, predicted) mean(abs((actual - predicted) / actual) * 100, na.rm = TRUE)

# ensure lengths align
ets_pred <- as.numeric(ets_forecast$mean)
arima_pred <- as.numeric(arima_forecast$mean)
actual <- as.numeric(test_ts)

metrics <- data.frame(
  Model = c("ETS", "ARIMA"),
  RMSE = c(rmse(actual, ets_pred), rmse(actual, arima_pred)),
  MAE  = c(mae(actual, ets_pred), mae(actual, arima_pred)),
  MAPE = c(mape(actual, ets_pred), mape(actual, arima_pred))
)

print(metrics)
write_csv(metrics, "data/forecast_accuracy_metrics.csv")

# -----------------------------
# 7. Save final forecast values (including prediction intervals)
# -----------------------------
# ETS results
ets_out <- data.frame(
  Date = seq(as.Date(time(ets_forecast$mean)[1]), by = "month", length.out = length(ets_forecast$mean)),
  Forecast = as.numeric(ets_forecast$mean),
  Lo80 = as.numeric(ets_forecast$lower[,1]),
  Hi80 = as.numeric(ets_forecast$upper[,1]),
  Lo95 = as.numeric(ets_forecast$lower[,2]),
  Hi95 = as.numeric(ets_forecast$upper[,2]),
  Model = "ETS"
)

# ARIMA results
arima_out <- data.frame(
  Date = seq(as.Date(time(arima_forecast$mean)[1]), by = "month", length.out = length(arima_forecast$mean)),
  Forecast = as.numeric(arima_forecast$mean),
  Lo80 = as.numeric(arima_forecast$lower[,1]),
  Hi80 = as.numeric(arima_forecast$upper[,1]),
  Lo95 = as.numeric(arima_forecast$lower[,2]),
  Hi95 = as.numeric(arima_forecast$upper[,2]),
  Model = "ARIMA"
)

write_csv(ets_out, "data/ets_forecast_next12.csv")
write_csv(arima_out, "data/arima_forecast_next12.csv")

# -----------------------------
# 8. Print a short summary conclusion
# -----------------------------
cat("\nForecast accuracy (lower is better):\n")
print(metrics)
best_model <- metrics$Model[which.min(metrics$RMSE)]
cat("\nBased on RMSE, best model:", best_model, "\n")
cat("Files saved in 'data/' and 'plots/' folders.\n")

########################################################################
# End of script
########################################################################

