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
data <- read_csv('q_normalized_data.csv')
data <- read_csv('q_normalized_data.csv')

# Extract data for each species
q_normalized_pions <- na.omit(data$q_normalized_pions)
q_normalized_kaons <- na.omit(data$q_normalized_kaons)
q_normalized_protons <- na.omit(data$q_normalized_protons)

# Define the Landau distribution fit function
fit_landau <- function(data) {
  fit <- fitdistr(data, densfun="landau")
  return(fit$estimate)
}

# Fit the Landau distribution and get the MPV
mpv_pions <- fit_landau(q_normalized_pions)
mpv_kaons <- fit_landau(q_normalized_kaons)
mpv_protons <- fit_landau(q_normalized_protons)

# Function to plot histogram and Landau fit
plot_hist_and_fit <- function(data, mpv, color, label) {
  ggplot(data, aes(x = data)) +
    geom_histogram(aes(y = ..density..), bins = 50, fill = color, alpha = 0.6) +
    stat_function(fun = function(x) dlandau(x, mpv), color = color, size = 1) +
    labs(title = paste('Landau fit for', label, '(MPV =', round(mpv, 2), ')'),
         x = 'Normalized $q_{mip}$', y = 'Density') +
    theme_minimal()
}

# Plot histograms and fits
plot_hist_and_fit(q_normalized_pions, mpv_pions, 'red', 'Pions')
plot_hist_and_fit(q_normalized_kaons, mpv_kaons, 'green', 'Kaons')
plot_hist_and_fit(q_normalized_protons, mpv_protons, 'blue', 'Protons')

