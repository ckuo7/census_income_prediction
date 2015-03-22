setwd("~/MSAN/621/final")

setwd('~/Desktop/module2/MSAN_621/project/')
# raw data
data <- read.csv('census-income.data', header=F)
ncol(data)
head(data)

# without target
new_data <- subset(data,select=seq(1,41))
new_data <- subset(new_data,select=-V25)
ncol(new_data)
head(new_data)

# trying to investigate correlation matrix between variables in raw data
cor(new_data)

# subset
#data <- data[sample(nrow(data), 500), ]
new_data$V42 <- factor(data$V42)
head(new_data)
str(new_data)

# factanal(data)

model <- glm(V42 ~ ., data=new_data, family="binomial")

library(MASS)
stepAIC(model)
stepAIC(model, direction='both')

install.packages('polycor')
library(polycor)
polycor(data)

library(psych)
install.packages('ICC')
library(ICC)
ICCest(data=data)