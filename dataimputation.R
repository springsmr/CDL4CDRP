#! /usr/bin/env Rscript
library(DMwR)
#多重插补法处理缺失，结果转存  
library(mice)
f <- function(x,y,z=1){
	result <- x+(2*y)+(3*z)
}
print(f(2,3,4))

knnImput <- function(filename){
	dataset <- read.table(filename,header=TRUE,sep=",")
	dataset_knn2 <- knnImputation(dataset[,1:ncol(dataset)-1],k=1)
	out <- paste("KNN1",filename,sep="_")
	write.csv(dataset_knn2,out)
	print(out)
}
miceImput <- function(filename,med="pmm"){
	dataset <- read.table(filename,header=TRUE,sep=",")
	mice_mod <- mice(dataset[,1:ncol(dataset)-1],method=med)
	miceOutput <- complete(mice_mod)
	out <- paste("mice",med,sep="_")
	out <- paste(out,filename,sep="_")
	write.csv(miceOutput,out)
	print(out)
}
chronicNames <- c("chronicNumric.csv","chronic_missing10Numric.csv","chronic_missing20Numric.csv","chronic_missing30Numric.csv")
dermatologyNames <- c("dermatologyNumric.csv","dermatology_missing10Numric.csv","dermatology_missing20Numric.csv","dermatology_missing30Numric.csv")

# for(filename in dermatologyNames) knnImput(filename)
# for(filename in chronicNames) knnImput(filename)
for(filename in dermatologyNames) miceImput(filename,"rf")
for(filename in chronicNames) miceImput(filename,"rf")
