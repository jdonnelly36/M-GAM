# Before running,
# set to appropriate working directory
# (exact setwd command ommitted, because it depends on R setup)

library("mice")
library("VIM")

num_imputations <- 20

df <- read.csv('fico_full.csv')

# Vanilla MICE does not differentiate between different types of missingness
# (to our knowledge)
# So replace all missingness encodings with NA
df <- replace(df, df < 0, NA)

#To start, we're doing imputation off of a semi-supervised learning paradigm
# so we want to impute just for all features except the target PoorRiskPerf
df_X <- df[, -24]
imp_X <- mice(df_X, m=num_imputations, seed = 2000)
#print(imp_X)
#complete(imp_X)

temp_df = complete(imp_X)
temp_df$PoorRiskPerformance = df$PoorRiskPerformance
write.csv(temp_df, 'fico_semi_supervised_imputation_1.csv', row.names=FALSE)

for (i in c(2:num_imputations)) {
  temp_df = complete(imp_X, i)
  temp_df$PoorRiskPerformance = df$PoorRiskPerformance
  write.csv(temp_df, paste('fico_semi_supervised_imputation_', i, '.csv', sep=''), row.names=FALSE)
}
#md.pattern(df)
#md.pairs(df)
#imp <- mice(df, seed = 2023)
#complete(imp)
#stripplot(imp, pch = 20, cex = 1.2)
#write.csv(complete(imp), 'fico_all_imputations.csv')

