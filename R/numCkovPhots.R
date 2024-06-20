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
data
# masks: 
pions<-data["pions"]
kaons<-data["kaons"]
protons<-data["protons"]




num_phots <-data["num_phots"]
# Filter out rows where num_phots is greater than 0
# Read the data from CSV
library(dplyr)  # For data manipulation
# Plot histogram of num_phots greater than 0 using ggplot2



# Filter the data for momentum range and ensure num_phots > 0
filtered_data <- data %>%
  filter(Momentum >= 2, Momentum <= 2.2, num_phots > 0)

# Create separate data frames for each species
pions_data <- filtered_data %>% filter(pions == TRUE)
kaons_data <- filtered_data %>% filter(kaons == TRUE)
protons_data <- filtered_data %>% filter(protons == TRUE)



# Combine data with an identifier column
combined_data <- bind_rows(
  pions_data %>% filter(ckov_recon > 0) %>% mutate(species = "Pions"),
  kaons_data %>% filter(ckov_recon > 0) %>% mutate(species = "Kaons"),
  protons_data %>% filter(ckov_recon > 0) %>% mutate(species = "Protons")
)

# Create a single plot with different colors
ggplot(combined_data, aes(x=ckov_recon, fill=species)) +
  geom_histogram(binwidth=0.001, alpha=0.6, position="identity") +  # 'identity' to overlay
  scale_fill_manual(values=c("blue", "red", "green")) +
  labs(x="ckov_recon [rad]", y="Frequency", title="Histogram of ckov_recon for Particles") +
  theme_minimal()





ggplot(pions_data %>% filter(ckov_recon > 0), aes(x=ckov_recon)) +
  geom_histogram(binwidth=0.01, fill="blue", color="black") +
  labs(x="ckov_recon [rad]", y="Frequency", title="Histogram of ckov_recon")


ggplot(kaons_data %>% filter(ckov_recon > 0), aes(x=ckov_recon)) +
  geom_histogram(binwidth=0.01, fill="blue", color="black") +
  labs(x="ckov_recon [rad]", y="Frequency", title="Histogram of ckov_recon")

ggplot(protons_data %>% filter(ckov_recon > 0), aes(x=ckov_recon)) +
  geom_histogram(binwidth=0.01, fill="blue", color="black") +
  labs(x="ckov_recon [rad]", y="Frequency", title="Histogram of ckov_recon")



# Print plots (In an interactive R session, these would render directly)
print(plot_num_phots)
print(plot_ckov_recon)





ckov_recon <-data["ckov_recon"]
num_phots <-data["num_phots"]

num_phots_filt <-num_phots[num_phots>0]
ckov_recon_filt <-ckov_recon[num_phots>0]

length(num_phots_filt)
length(ckov_recon_filt)
filtered_data <- filter(data, num_phots > 0)

sin_qs_theta_c <-sin(ckov_recon_filt)^2
length(sin_qs_theta_c)




# HeatMap of CkovRecon vs numPhotons --------------------------------------



# 2D histogram of ckov_recon vs num_phots
ggplot(filtered_data, aes(x=ckov_recon, y=num_phots)) +
  geom_bin2d(bins=30, aes(fill=..count..)) +  # Adjust 'bins' as needed
  scale_fill_viridis_c(option="C") +  # Use Viridis color scale
  labs(x="ckov_recon [rad]", y="Number of Cherenkov Photons", title="2D Histogram of ckov_recon vs num_phots")




# 2D histogram with sin^2(ckov_recon) on x-axis for filtered data
ggplot(filtered_data, aes(x=sin(ckov_recon)^2, y=num_phots)) +
  geom_bin2d(bins=30, aes(fill=..count..)) +
  scale_fill_viridis_c() +  # Viridis color map
  labs(x="sin^2(ckov_recon) [rad^2]", y="Number of Cherenkov Photons", title="2D Histogram of sin^2(ckov_recon) vs num_phots")


# Load necessary libraries
library(ggplot2)
library(dplyr)


# Create a new variable for binned sin^2(ckov_recon)
filtered_data <- filtered_data %>%
  mutate(bin_sin2_ckov_recon = cut(sin(ckov_recon)^2, breaks=30))  # Adjust the number of breaks as needed

# Plotting the 2D histogram with trend line of mean values
ggplot(filtered_data, aes(x=sin(ckov_recon)^2, y=num_phots)) +
  geom_bin2d(bins=30, aes(fill=..count..)) +  # This shows the 2D histogram
  stat_summary_bin(
    fun.y = "mean", geom = "line", bins = 30, aes(group=1, color="Mean num_phots"),
    size=1.5  # Adjust the line thickness as needed
  ) +
  scale_fill_viridis_c() +  # Viridis color map
  labs(x="sin^2(ckov_recon) [rad^2]", y="Number of Cherenkov Photons", title="2D Histogram of sin^2(ckov_recon) vs num_phots") +
  theme_minimal() +  # Cleaner theme
  theme(legend.position="right")  # Adjust legend position

#lm_model <- lm(Theta_P[protons] ~ Momentum[protons], data=data)




# Perform linear regression
lm_model <- lm(Theta_P ~ Momentum, data=data)

# Print summary of the linear regression model
summary(lm_model)

# Predict Theta_P using the regression model
predicted_theta_p <- predict(lm_model, newdata=data)

# Read the data from CSV

# Scatter plot using ggplot2
ggplot(data, aes(x=ckov_recon, y=num_phots)) +
  geom_point() +
  labs(x="ckov_recon [rad]", y="Number of cherenkov Photons", title="Scatter Plot of ckov_recon vs num_phots")


# Scatter plot using ggplot2 with sin^2(ckov_recon) on x-axis
ggplot(data, aes(x=sin(ckov_recon)^2, y=num_phots)) +
  geom_point() +
  labs(x="sin^2(ckov_recon) [rad^2]", y="Number of Cherenkov Photons", title="Scatter Plot of sin^2(ckov_recon) vs num_phots")


# Scatter plot using ggplot2
ggplot(data, aes(x=Momentum, y=num_phots)) +
  geom_point() +
  labs(x="Momentum (GeV/c)", y="Number of cherenkov Photons", title="Scatter Plot of ckov_recon vs num_phots")

