#!/usr/bin/Rscript
library(optparse)
library(mice)
library(stringr)
library(tidyverse)

# Define the impmutation function
MICE_impute <- function( train, val, test,
                         output_dir,
                         random_state = 0,
                         bin_feat = NULL,
                         ord_feat = NULL,
			 max.it = 5 ){
    # max.it: the maximum number of iteration given to MICE and mice.reuse function

    # Load data from csv files, remove the final column which contains
    # the outcome variable.
    train <- read.csv(file = train)
    train[length(train)] <- NULL
    val <- read.csv(file = val)
    val[length(val)] <- NULL
    test <- read.csv(file = test)
    test[length(test)] <- NULL

    # from https://raw.githubusercontent.com/prockenschaub/Misc/master/R/mice.reuse.R
    source("mice.reuse.R")
    if (!is.null(bin_feat) | !is.null(ord_feat)){
      fac_ind <- c(bin_feat, ord_feat)
      train_fct <- train %>% mutate_at(fac_ind, factor)
      val_fct <- val %>% mutate_at(fac_ind, factor)
      test_fct <- test %>% mutate_at(fac_ind, factor)
      n_col <- ncol(train)
      meth <- rep("pmm", n_col)
      meth[bin_feat] = "logreg"
      meth[ord_feat] = "polr"
      imp <- mice(train_fct,  m=1, maxit = max.it, seed = random_state, method = meth,
                  printFlag = FALSE)
      val_imp <- mice.reuse(imp, val_fct, maxit = max.it, printFlag = FALSE)
      test_imp <- mice.reuse(imp, test_fct, maxit = max.it, printFlag = FALSE)
    } else {
      imp <- mice(train, m=1, maxit = max.it, seed = random_state, printFlag = FALSE)
      val_imp <- mice.reuse(imp, val, maxit = max.it, printFlag = FALSE)
      test_imp <- mice.reuse(imp, test, maxit = max.it, printFlag = FALSE)
    }

    if (!dir.exists(output_dir)){
      dir.create(output_dir, showWarnings = TRUE, recursive = TRUE)
    }
    write_csv(complete(imp), file=file.path(output_dir, "imputed_train_x.csv"))
    write_csv(val_imp[[1]], file=file.path(output_dir, "imputed_val_x.csv"))
    write_csv(test_imp[[1]], file=file.path(output_dir, "imputed_test_x.csv"))
}

# Argument Parsing

parse_ind <- function(string){
  if (is.null(string)) {
    return(NULL)
  }
  list = strsplit(string, "-")
  return(as.integer(unlist(list)))
}

option_list = list(
  make_option(c("--train"), type="character", default=NULL,
              help="training dataset file path", metavar="character"),
  make_option(c("--val"), type="character", default=NULL,
              help="validation dataset file path", metavar="character"),
  make_option(c("--test"), type="character", default=NULL,
              help="test dataset file path", metavar="character"),
  make_option(c("--outdir"), type="character", default=NULL,
              help="output directory", metavar="character"),
  make_option(c("--bin_feat"), type="character", default=NULL,
              help="indices of binary features",metavar="character"),
  make_option(c("--ord_feat"), type="character", default=NULL,
              help="indices of ordinal features", metavar="character"),
  make_option(c("--random_state"), type="integer", default=NULL,
              help="random state for mice", metavar="integer")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

MICE_impute(train=opt$train,
            val=opt$val,
            test=opt$test,
            output_dir=opt$outdir,
            random_state=opt$random_state,
            bin_feat=parse_ind(opt$bin_feat),
            ord_feat=parse_ind(opt$ord_feat))
