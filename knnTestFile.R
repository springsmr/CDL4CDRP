#! /usr/bin/env Rscript
library(DMwR)
library(mice)
chronicN <-"/home/min/bigdata/data/ex1/chronicNumric.csv"
datasetN <- read.table(chronicN,header=TRUE,sep=",")
print(datasetN)
#datasetN_knn5 <- knnImputation(datasetN[,1:24],k=5)
#write.csv(datasetN_knn5,"/home/min/bigdata/data/ex1/chronicNumricKNN5.csv")
