install.packages("nnet")
install.packages('lme4')
install.packages("mixcat")
install.packages("devtools")
install.packages("effects")
install.packages("sjPlot")
install.packages("jtools")
install.packages("dplyr")
library(ggplot2); theme_set(theme_bw())
library(nnet)
library(lme4)
library(effects)
library(ggplot2)
library(sjPlot)
library(sjmisc)
library(sjlabelled)
library(jtools)
library(dplyr)

#importDataFrame
df_MultinomialLogisticRegression <- read.csv('/Users/fatma/Desktop/PlanningCode/modeling /DataFrameMultinomialRegression.csv', header = TRUE)

#multinomialLogisticRegression

#Convert factors to factors
df_MultinomialLogisticRegression$order <- as.factor(df_MultinomialLogisticRegression$order)

df_MultinomialLogisticRegression$participant<-as.factor(df_MultinomialLogisticRegression$participant)



# Initialize an empty list to store results
results_multinomial <- list()

# Loop over each unique participant
for (subject in unique(df_MultinomialLogisticRegression$participant)) {
  # Subset the data for the current subject
  subject_data <- df_MultinomialLogisticRegression %>% filter(participant == subject)
  
  # Fit the multinomial logistic regression model
  model <- multinom(order ~ imp_category + diff_category, data = subject_data)
  
  # Extract summary of the model
  model_summary <- summary(model)
  
  # Coefficients matrix
  coefs <- model_summary$coefficients
  
  # Standard errors matrix
  se <- model_summary$standard.errors
  
  # Extract coefficients and standard errors for each outcome level (2, 3 assumed)
  results_multinomial[[as.character(subject)]] <- data.frame(
    participant = rep(subject, 2),
    level = c("2", "3"),
    intercept = c(coefs["2", "(Intercept)"], coefs["3", "(Intercept)"]),
    imp_category_coef = c(coefs["2", "imp_category"], coefs["3", "imp_category"]),
    diff_category_coef = c(coefs["2", "diff_category"], coefs["3", "diff_category"]),
    intercept_se = c(se["2", "(Intercept)"], se["3", "(Intercept)"]),
    imp_category_se = c(se["2", "imp_category"], se["3", "imp_category"]),
    diff_category_se = c(se["2", "diff_category"], se["3", "diff_category"]),
    t_intercept = c(coefs["2", "(Intercept)"]/se["2", "(Intercept)"], coefs["3", "(Intercept)"]/se["3", "(Intercept)"]),
    t_imp_category_coef = c(coefs["2", "imp_category"]/se["2", "imp_category"], coefs["3", "imp_category"]/se["3", "imp_category"]),
    t_diff_category_coef = c(coefs["2", "diff_category"]/se["2", "diff_category"], coefs["3", "diff_category"]/se["3", "diff_category"])
  )
}

# Convert list of data frames to a single data frame
results_df <- do.call(rbind, results_multinomial)

# Print the results to confirm
print(results_df)

# Combine all results into a single data frame
results_df <- do.call(rbind, results_multinomial)

results_multinomial <- results_df %>%
  mutate(across(where(is.numeric), round, 3))

# Print the results to confirm
print(results_multinomial)
# Assuming results_df is the data frame obtained from the loop
# We will calculate the population-level t-values and p-values

# Aggregate mean and standard error for each coefficient over all participants
aggregate_results <- results_df %>%
  group_by(level) %>%
  summarise(
    mean_intercept = mean(intercept, na.rm = TRUE),
    se_intercept = sd(intercept, na.rm = TRUE) / sqrt(n()),
    mean_imp_category_coef = mean(imp_category_coef, na.rm = TRUE),
    se_imp_category_coef = sd(imp_category_coef, na.rm = TRUE) / sqrt(n()),
    mean_diff_category_coef = mean(diff_category_coef, na.rm = TRUE),
    se_diff_category_coef = sd(diff_category_coef, na.rm = TRUE) / sqrt(n())
  )

# Calculate t-values and p-values for the aggregated results
aggregate_results <- aggregate_results %>%
  mutate(
    t_intercept = mean_intercept / se_intercept,
    t_imp_category_coef = mean_imp_category_coef / se_imp_category_coef,
    t_diff_category_coef = mean_diff_category_coef / se_diff_category_coef,
    
    # Correctly calculate p-values using the t-distribution with 27 degrees of freedom
    p_intercept = 2 * (1 - pt(abs(t_intercept), 27)),
    p_imp_category_coef = 2 * (1 - pt(abs(t_imp_category_coef), 27)),
    p_diff_category_coef = 2 * (1 - pt(abs(t_diff_category_coef), 27))
  )


# Print the aggregated results
print(aggregate_results)

file_path <- "~/Desktop/aggregate_results_thenewone.csv"


# Save the results to a CSV file
write.csv(aggregate_results, file_path, row.names = FALSE)
# Filter data for level 2
# Filter data and round to 4 digits
level_2_data <- results_df %>%
  filter(level == 2) %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))


# Filter data for level 3
# Filter data and round to 4 digits
level_3_data <- results_df %>%
  filter(level == 3) %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))

# Calculate the mean of imp_category_coef for level 2
mean_imp_category_coef_level_2 <- mean(level_2_data$imp_category_coef)

# Calculate the mean of imp_category_coef for level 3
mean_imp_category_coef_level_3 <- mean(level_3_data$imp_category_coef)
# Calculate the mean of imp_category_coef for level 2
mean_diff_category_coef_level_2 <- mean(level_2_data$diff_category_coef)

# Calculate the mean of imp_category_coef for level 3
mean_diff_category_coef_level_3 <- mean(level_3_data$diff_category_coef)

mean_t_imp_category_coef_level_3 <- mean(level_3_data$t_imp_category_coef)
# Print the results
cat("Mean of imp_category_coef for level 2:", mean_imp_category_coef_level_2, "\n")
cat("Mean of imp_category_coef for level 3:", mean_imp_category_coef_level_3, "\n")



file_path <- "~/Desktop/results_multinomial.csv"

file_path_1 <- "~/Desktop/level_2_data.csv"
file_path_2 <- "~/Desktop/level_3_data.csv"
# Save the results to a CSV file
write.csv(results_multinomial, file_path, row.names = FALSE)

write.csv(level_2_data, file_path_1, row.names = FALSE)
write.csv(level_3_data, file_path_2, row.names = FALSE)
# Print the results to confirm
print(results_df)
# Extract the coefficients for imp_category and diff_category for each subject
imp_category_betas <- results_df$imp_category_coef
diff_category_betas <- results_df$diff_category_coef
intercept_all<-results_df$intercept
# Perform a paired t-test
t_test_result <- t.test(imp_category_betas, diff_category_betas, paired = TRUE)

# Print the result
print(t_test_result)
#RTDiffcultyAndimportance
# Convert factors to factors
file1$order <- as.factor(file1$order)
file1$participant <- as.factor(file1$participant)
file2_notnormalizeddifficultyandimportance$order<-as.factor(file2_notnormalizeddifficultyandimportance$order)
file2_notnormalizeddifficultyandimportance$participant<-as.factor(file2_notnormalizeddifficultyandimportance$participant)

# Initialize an empty list to store results
results_ReactionTime<- list()


# Loop over each unique participant
for (subject in unique(df_MultinomialLogisticRegression$participant)) {
  # Debugging: Print the current subject
  print(subject)
  
  # Subset the data for the current subject
  subject_data <- df_MultinomialLogisticRegression %>% filter(participant == subject)
  
  # Try to fit the linear model and extract coefficients
  tryCatch({
    model <- lm(log(RT) ~ imp_category + diff_category, data = subject_data)
    
    # Extract summary of the model
    model_summary <- summary(model)
    
    # Coefficients matrix
    coefs <- model_summary$coefficients
    
    # Standard errors matrix (use column 2 for standard errors)
    se <- coef(model_summary)[, "Std. Error"]
    
    # Ensure all necessary coefficients are present
    required_coefs <- c("(Intercept)", "imp_category", "diff_category")
    if (all(required_coefs %in% rownames(coefs))) {
      # Extract coefficients and standard errors for the model
      results_ReactionTime[[as.character(subject)]] <- data.frame(
        participant = subject,
        intercept = coefs["(Intercept)", "Estimate"],
        imp_category_coef = coefs["imp_category", "Estimate"],
        diff_category_coef = coefs["diff_category", "Estimate"],
        intercept_se = se["(Intercept)"],
        imp_category_se = se["imp_category"],
        diff_category_se = se["diff_category"],
        t_intercept = coefs["(Intercept)", "Estimate"] / se["(Intercept)"],
        t_imp_category_coef = coefs["imp_category", "Estimate"] / se["imp_category"],
        t_diff_category_coef = coefs["diff_category", "Estimate"] / se["diff_category"]
      )
    } else {
      print(paste("Missing coefficients for participant", subject))
    }
  }, error = function(e) {
    print(paste("Error fitting model for participant", subject, ":", e$message))
  })
}

# Convert list of data frames to a single data frame
results_df <- do.call(rbind, results_ReactionTime)


# Degrees of freedom
df <- n - 1

results_ReactionTime <- results_df %>%
  mutate(across(where(is.numeric), round, 3))

# Print the results to confirm
print(results_ReactionTime)


file_path <- "~/Desktop/results_ReactionTime.csv"


# Save the results to a CSV file
write.csv(results_ReactionTime, file_path, row.names = FALSE)
# Aggregate mean and standard error for each coefficient over all participants
aggregate_results <- results_df %>%
  summarise(
    mean_intercept = mean(intercept, na.rm = TRUE),
    se_intercept = sd(intercept, na.rm = TRUE) / sqrt(n()),
    mean_imp_category_coef = mean(imp_category_coef, na.rm = TRUE),
    se_imp_category_coef = sd(imp_category_coef, na.rm = TRUE) / sqrt(n()),
    mean_diff_category_coef = mean(diff_category_coef, na.rm = TRUE),
    se_diff_category_coef = sd(diff_category_coef, na.rm = TRUE) / sqrt(n())
  )

aggregate_results <- aggregate_results %>%
  mutate(
    # Calculate t-values
    t_intercept = mean_intercept / se_intercept,
    t_imp_category_coef = mean_imp_category_coef / se_imp_category_coef,
    t_diff_category_coef = mean_diff_category_coef / se_diff_category_coef,
    
    # Calculate p-values for the t-values using the t-distribution
    p_intercept = 2 * (1 - pt(abs(t_intercept), df = 27)),
    p_imp_category_coef = 2 * (1 - pt(abs(t_imp_category_coef), df = 27)),
    p_diff_category_coef = 2 * (1 - pt(abs(t_diff_category_coef), df = 27)))



# Extract the coefficients for imp_category and diff_category for each subject
imp_category_betas <- results_df$imp_category_coef
diff_category_betas <- results_df$diff_category_coef
intercept_all<-results_df$intercept
# Perform a paired t-test
t_test_result <- t.test(imp_category_betas,diff_category_betas, paired = TRUE)

# Print the result
print(t_test_result)

#ConfidenceModelWithDifficultyAndImportanceRTAndGazeswitch
#importDataFrame
df_Confidece <- read.csv('/Users/fatma/Desktop/PlanningCode/modeling /DataFrameConfidenceModel.csv', header = TRUE)

df_Confidece$participant <- as.factor(df_Confidece$participant)
# Initialize an empty list to store results
results_confidence <- list()

# Loop over each unique participant
for (subject in unique(file2$participant)) {
  # Debugging: Print the current subject
  print(subject)
  
  # Subset the data for the current subject
  subject_data <- file2 %>% filter(participant == subject)
  
  # Try to fit the linear model and extract coefficients
  tryCatch({
    model <- lm(slider_confidence_response ~ imp_category_mean + diff_category_mean + log(RTTotal) + W_B_total, data = subject_data)
    
    # Extract summary of the model
    model_summary <- summary(model)
    
    # Coefficients matrix
    coefs <- coef(model_summary)
    
    # Standard errors matrix (use column 2 for standard errors)
    se <- coef(model_summary)[, "Std. Error"]
    
    # Ensure all necessary coefficients are present
    required_coefs <- c("(Intercept)", "imp_category_mean", "diff_category_mean", "log(RTTotal)", "W_B_total")
    if (all(required_coefs %in% rownames(coefs))) {
      # Extract coefficients and standard errors for the model
      results_confidence[[as.character(subject)]] <- data.frame(
        participant = subject,
        intercept = coefs["(Intercept)", "Estimate"],
        imp_category_coef = coefs["imp_category_mean", "Estimate"],
        diff_category_coef = coefs["diff_category_mean", "Estimate"],
        log_RTTotal_coef = coefs["log(RTTotal)", "Estimate"],
        W_B_total_coef = coefs["W_B_total", "Estimate"],
        intercept_se = se["(Intercept)"],
        imp_category_se = se["imp_category_mean"],
        diff_category_se = se["diff_category_mean"],
        log_RTTotal_se = se["log(RTTotal)"],
        W_B_total_se = se["W_B_total"],
        t_intercept = coefs["(Intercept)", "Estimate"] / se["(Intercept)"],
        t_imp_category_coef = coefs["imp_category_mean", "Estimate"] / se["imp_category_mean"],
        t_diff_category_coef = coefs["diff_category_mean", "Estimate"] / se["diff_category_mean"],
        t_log_RTTotal_coef = coefs["log(RTTotal)", "Estimate"] / se["log(RTTotal)"],
        t_W_B_total_coef = coefs["W_B_total", "Estimate"] / se["W_B_total"]
      )
    }
  }, error = function(e) {
    # Handle errors (e.g., model fitting issues) here if needed
    print(paste("Error with subject:", subject))
  })
}

# Convert list of data frames to a single data frame
results_df <- do.call(rbind, results_confidence)
aggregate_results <- results_df %>%
  summarise(
    mean_intercept = mean(intercept, na.rm = TRUE),
    se_intercept = sd(intercept, na.rm = TRUE) / sqrt(n()),
    
    mean_imp_category_coef = mean(imp_category_coef, na.rm = TRUE),
    se_imp_category_coef = sd(imp_category_coef, na.rm = TRUE) / sqrt(n()),
    
    mean_diff_category_coef = mean(diff_category_coef, na.rm = TRUE),
    se_diff_category_coef = sd(diff_category_coef, na.rm = TRUE) / sqrt(n()),
    
    mean_log_RTTotal_coef = mean(log_RTTotal_coef, na.rm = TRUE),
    se_log_RTTotal_coef = sd(log_RTTotal_coef, na.rm = TRUE) / sqrt(n()),
    
    mean_W_B_total_coef = mean(W_B_total_coef, na.rm = TRUE),   # Corrected line
    se_W_B_total_coef = sd(W_B_total_coef, na.rm = TRUE) / sqrt(n())  # Added standard error for W_B_total_coef
  )

# Print the results
print(aggregate_results)

aggregate_results <- aggregate_results %>%
  mutate(
    # Calculate t-values
    t_intercept = mean_intercept / se_intercept,
    t_imp_category_coef = mean_imp_category_coef / se_imp_category_coef,
    t_diff_category_coef = mean_diff_category_coef / se_diff_category_coef,
    
    # Calculate p-values for the t-values using the t-distribution
    p_intercept = 2 * (1 - pt(abs(t_intercept), df = 27)),
    p_imp_category_coef = 2 * (1 - pt(abs(t_imp_category_coef), df = 27)),
    p_diff_category_coef = 2 * (1 - pt(abs(t_diff_category_coef), df = 27)),
    
    # Calculate t-value and p-value for log_RTTotal
    t_log_RTTotal_coef = mean_log_RTTotal_coef / se_log_RTTotal_coef,
    p_log_RTTotal_coef = 2 * (1 - pt(abs(t_log_RTTotal_coef), df = 27)),
    
    # Calculate t-value and p-value for W_B_total
    t_W_B_total_coef = mean_W_B_total_coef / se_W_B_total_coef,
    p_W_B_total_coef = 2 * (1 - pt(abs(t_W_B_total_coef), df = 27))
  )

# Print the results
print(aggregate_results)


# Print the results
print(aggregate_results)

# Print the results to confirm
print(results_df)




# Print the results to confirm
print(results_df)
results_confidence_all <- results_df %>%
  mutate(across(where(is.numeric), round, 3))

# Print the results to confirm
print(results_confidence_all)


file_path <- "~/Desktop/results_confidence_all.csv"


# Save the results to a CSV file
write.csv(results_confidence_all, file_path, row.names = FALSE)

