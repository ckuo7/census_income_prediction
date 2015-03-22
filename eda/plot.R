
library('rattle')
library('rpart')
library('randomForest')
setwd('~/Desktop/module2/MSAN_621/project//')

df <- read.csv('./census-income.data',header=F)

form <- formula(V42 ~ . - V25)
model <- randomForest(formula=form,data=df)
pdf('./rplot.pdf')
varImpPlot(model,main="Variable Importance Plot")
dev.off()

# training size vs testing accuracy
# SVC C = 10, gamma = 0.01

train <- c(5,20,50,100)
a <- c(0.846,0.8512,0.8541,0.8564)
r <- c(0.9043,0.8986,0.8986,0.8991)
p <- c(0.2702,0.2811,0.2852,0.2887)

par(mar=c(5, 4, 4, 4))
train <- c(5,20,50,100,186)
p2 <- c(0.2643, 0.2739,0.2765,0.2784,0.2814)
plot(train,p2,xlab='percent of training set size',ylab='precision',type='l')


pdf('./report/plot1.pdf')
ratio <- c(1,6/4,7/3,8/2,10/1)
p3<-c(0.8403,0.7772,0.6849,0.565, 0.349)
plot(ratio,p3,xlab='ratio',ylab='precision',type='l',col='red',
     main="Precision vs Ratio of the negative examples to the positive")
legend(x=c('topright'),legend=c('SVC with C=10\n and gamma = 0.01'),inset =0.01,col='red',lty=1,lw=2)
dev.off()

pdf('./report/plot2.pdf')
ratio <- c(1.5,2.33,4,5,5.3,5.5,5.6,5.7,5.8,6,7,10.0,15.0)
f <- c(0.466,0.5179,0.5675,0.575,0.5772,0.5783,0.5738,0.5781,0.5781,0.5735,0.5709,0.5183,0.4005)
#a <- c(0.8795,0.9088,0.9384,0.9442,0.9467,0.9453,0.9475,0.9477,0.9479,0.948,0.9514,0.9532,0.9503)
plot(ratio,f,type='l',col='red',ylab='F1-scores',
     xlab='ratio of negative training examples to the positive',
     main='Fig.2 F1-score vs ratio of negative training examples to the positive ')
legend(x=c('topright'),legend=c('SVC with C=10\n and gamma=0.01'),inset=0.01,col='red',lty=2,lw=2)
#lines(a)
dev.off()


pdf('./report/plot3.pdf')
#par(mar=c(5.1, 4.1, 4.1, 9.1), xpd=TRUE)
train <- c(5,20,50,100)
a <- c(0.846,0.8512,0.8541,0.8564)
r <- c(0.9043,0.8986,0.8986,0.8991)
p <- c(0.2702,0.2811,0.2852,0.2887)
plot(train,a, 
     xlab='Percent of training set',ylab='%',ylim=range(0,1),type='l',
     main="Accuracy vs Percent of training set")
lines(train,r,col='red',lw=1.5)
lines(train,p,col='blue',lw=1.5)
legend('bottomright',legend=c('Accuracy','Recall','Precision'),
       col=c('black','red','blue'),lty=1,horiz=TRUE)
dev.off()

