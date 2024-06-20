#install.packages("landau")
#library(landau)

install.packages("ggplot2")
install.packages("readr")
install.packages("MASS")

# Load necessary libraries
library(ggplot2)
library(readr)
library(MASS) # For fitting distributions

# Read the data from CSV
data <- read_csv('track_infos.csv')

# masks: 
pions<-data["pions"]
kaons<-data["kaons"]
protons<-data["protons"]

install.packages("ggplot2")
install.packages("readr")
install.packages("MASS")



lm_model <- lm(Theta_P[protons] ~ Momentum[protons], data=data)


# Load necessary libraries
library(ggplot2)
library(readr)
library(MASS) # For fitting distributions

# Read the data from CSV
data <- read_csv('track_infos.csv')



# Linear regression first degree ------------------------------------------

# Perform linear regression
lm_model <- lm(Theta_P ~ Momentum, data=data)

# Print summary of the linear regression model
summary(lm_model)

# Predict Theta_P using the regression model
predicted_theta_p <- predict(lm_model, newdata=data)

# Plot the original data and regression line
ggplot(data, aes(x=Momentum, y=Theta_P)) +
  geom_point() +
  geom_smooth(method="lm", se=FALSE, color="orange") +
  labs(x="Momentum (GeV/c)", y="Theta_P (radians)", title="Linear Regression of Theta_P vs Momentum")



# Logistic regression ------------------------------------------

library(minpack.lm)

# Define the logistic function with a negative growth rate
logistic_function <- function(x, x0, k, L, y0) {
  y0 + L / (1 + exp(-k * (x - x0)))
}

# Non-linear model fitting using minpack.lm for more robust fitting
nls_model <- nlsLM(Theta_P ~ logistic_function(Momentum, x0, k, L, y0),
                   data = data,
                   start = list(x0 = median(data$Momentum), k = -0.1, L = max(data$Theta_P) - min(data$Theta_P), y0 = max(data$Theta_P)),
                   control = nls.lm.control(maxiter = 200))

# Check the summary to see if the fit has improved
summary(nls_model)

# Plot the fit to see how well it matches the data
new_data <- data.frame(Momentum = seq(min(data$Momentum), max(data$Momentum), length.out = 300))
new_data$Theta_P <- predict(nls_model, newdata = new_data)

ggplot(data, aes(x = Momentum, y = Theta_P)) +
  geom_point() +
  geom_line(data = new_data, aes(x = Momentum, y = Theta_P), color = "red") +
  labs(title = "Logistic Fit of Theta_P vs Momentum", x = "Momentum (GeV/c)", y = "Theta_P (radians)")

# Calculate residuals for the logistic model
residuals_logistic <- residuals(nls_model)

# Basic plot of residuals
plot(residuals_logistic, main="Residuals of Logistic Model", ylab="Residuals", xlab="Index")

# Using ggplot2 to plot residuals
data$Residuals_Logistic <- residuals_logistic
ggplot(data, aes(x = seq_along(Residuals_Logistic), y = Residuals_Logistic)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype="dashed", color="red") +
  labs(title="Residuals of Logistic Model", x="Index", y="Residuals")


# Q-Q plot for logistic model residuals
qqnorm(residuals_logistic)
qqline(residuals_logistic, col="red")

# Using ggplot2 for Q-Q plot
ggplot(data, aes(sample = Residuals_Logistic)) +
  stat_qq() +
  stat_qq_line() +
  labs(title="Q-Q Plot of Logistic Model Residuals")



# Linear regression different polynomials ------------------------------------------

#### Polynomials
# Basic plot of residuals
par(mfrow=c(1,2))
plot(data$Residuals_Poly_2, main="Residuals of 2nd Degree Polynomial", ylab="Residuals", xlab="Index")
plot(data$Residuals_Poly_3, main="Residuals of 3rd Degree Polynomial", ylab="Residuals", xlab="Index")

# Using ggplot2 to plot residuals
ggplot(data) +
  geom_point(aes(x = seq_along(Residuals_Poly_2), y = Residuals_Poly_2), color="blue") +
  geom_hline(yintercept = 0, linetype="dashed", color="red") +
  labs(title="Residuals of 2nd Degree Polynomial", x="Index", y="Residuals") +
  theme_minimal() +
  facet_wrap(~Poly_Degree, scales="free_y")

# Fit a 2nd degree polynomial model
poly_model_2 <- lm(Theta_P ~ poly(Momentum, 2), data=data)

# Fit a 3rd degree polynomial model
poly_model_3 <- lm(Theta_P ~ poly(Momentum, 3), data=data)
# Assuming the necessary libraries are already loaded and the data is read
# Calculate residuals for polynomial models
data$Residuals_Poly_2 <- residuals(poly_model_2)
data$Residuals_Poly_3 <- residuals(poly_model_3)
# Summary of the models (optional, for detailed statistical output)
summary(poly_model_2)
summary(poly_model_3)
library(ggplot2)

# Create a data frame for predictions
predict_data <- data.frame(Momentum = seq(min(data$Momentum), max(data$Momentum), length.out = 300))
predict_data$Theta_P_2 <- predict(poly_model_2, newdata = predict_data)
predict_data$Theta_P_3 <- predict(poly_model_3, newdata = predict_data)

# Plot with ggplot2
ggplot(data, aes(x = Momentum, y = Theta_P)) +
  geom_point(aes(color = "Data Points"), alpha = 0.5) +  # Plot the raw data points
  geom_line(data = predict_data, aes(x = Momentum, y = Theta_P_2, color = "2nd Degree Polynomial")) +  # Add 2nd degree polynomial line
  geom_line(data = predict_data, aes(x = Momentum, y = Theta_P_3, color = "3rd Degree Polynomial"), linetype = "dashed") +  # Add 3rd degree polynomial line
  labs(title = "Polynomial Regression of Theta_P vs. Momentum",
       x = "Momentum (GeV/c)", y = "Theta_P (radians)",
       color = "Model Type") +
  scale_color_manual(values = c("Data Points" = "black", "2nd Degree Polynomial" = "blue", "3rd Degree Polynomial" = "red")) +
  theme_minimal()

predict_data_long <- rbind(
  transform(predict_data, Theta_P = Theta_P_2, Poly_Degree = Poly_Degree_2),
  transform(predict_data, Theta_P = Theta_P_3, Poly_Degree = Poly_Degree_3)
)

# Plot with ggplot2
ggplot(data, aes(x = Momentum, y = Theta_P)) +
  geom_point(aes(color = "Data Points"), alpha = 0.5) +  # Plot the raw data points
  geom_line(data = predict_data_long, aes(x = Momentum, y = Theta_P, color = Poly_Degree), linetype = "dashed") +  # Add polynomial lines
  labs(title = "Polynomial Regression of Theta_P vs. Momentum",
       x = "Momentum (GeV/c)", y = "Theta_P (radians)",
       color = "Model Type") +
  scale_color_manual(values = c("Data Points" = "black", "2nd Degree Polynomial" = "blue", "3rd Degree Polynomial" = "red")) +
  theme_minimal() +
  facet_wrap(~Poly_Degree, scales="free_y")
