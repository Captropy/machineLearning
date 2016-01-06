# This is my new script for the machine learning course

#++++++++++++++++++++
# Read data
#++++++++++++++++++++

pml.training <- read.csv("pml-training.csv")
pml.testing <- read.csv("pml-testing.csv")

#++++++++++++++++++++
# Split training in test and cross-validation
#++++++++++++++++++++

## 75% of the sample size
smp_size <- floor(0.75 * nrow(pml.training))

## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(pml.training)), size = smp_size)

train <- pml.training[train_ind, ]
test <- pml.training[-train_ind, ]
