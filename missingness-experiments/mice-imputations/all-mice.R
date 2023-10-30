# Before running,
# set to appropriate working directory
# (exact setwd command ommitted, because it depends on R setup)

library("mice")
library("VIM")

num_imputations <- 100
num_trials <- 5
set.seed(2023)
accuracies <- c()
times <- c()

df <- read.csv('fico_full.csv')
#take only a small subset of high missingness features
df <- df[, c(16,20,21,22,23,24)]

for (j in c(1:num_trials)){
  test_indices = sample(nrow(df), size=0.2*nrow(df))

  # Vanilla MICE does not differentiate between different types of missingness
  # (to our knowledge)
  # So replace all missingness encodings with NA
  df <- replace(df, df < 0, NA)
  
  #create semi-supervised training set: missing values for Y_test
  df_train <- df
  df_train$PoorRiskPerformance <- replace(df_train$PoorRiskPerformance, test_indices, NA)
  
  time <- system.time(imp <- mice(df_train, m=num_imputations))
  times <- append(times, time[['sys.self']] + time[['user.self']])
  
  temp_df = complete(imp)
  
  #track votes from each imputation
  votes_for_label_1 = temp_df[test_indices, "PoorRiskPerformance"]
  
  for (i in c(2:num_imputations)) {
    temp_df = complete(imp, i)
    votes_for_label_1 = votes_for_label_1 + temp_df[test_indices, "PoorRiskPerformance"]
  }
  
  majority_vote = votes_for_label_1 > num_imputations/2
  accuracies <- append(accuracies, mean(majority_vote == df[test_indices, "PoorRiskPerformance"]))
}

print(accuracies)
print(times)
