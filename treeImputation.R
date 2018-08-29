#! /usr/bin/env Rscript
library(rpart)
knnImput <- function(filename){
	dataset <- read.table(filename,header=TRUE,sep=",")
	
class_mod <- rpart(rad ~ . - medv, data=BostonHousing[!is.na(dataset[c("sg", "al", "s", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane")]), ], method="class", na.action=na.omit)  # 因为rad是因子
anova_mod <- rpart(ptratio ~ . - medv, data=BostonHousing[!is.na(BostonHousing$ptratio), ], method="anova", na.action=na.omit)  # ptratio是数值变量
rad_pred <- predict(class_mod, BostonHousing[is.na(BostonHousing$rad), ])
ptratio_pred <- predict(anova_mod, BostonHousing[is.na(BostonHousing$ptratio), ])
	dataset_knn2 <- knnImputation(dataset[,1:ncol(dataset)-1],k=1)
	out <- paste("KNN1",filename,sep="_")
	write.csv(dataset_knn2,out)
	print(out)
}

class_mod <- rpart(rad ~ . - medv, data=BostonHousing[!is.na(t), ], method="class", na.action=na.omit)  # 因为rad是因子
anova_mod <- rpart(ptratio ~ . - medv, data=BostonHousing[!is.na(BostonHousing$ptratio), ], method="anova", na.action=na.omit)  # ptratio是数值变量
rad_pred <- predict(class_mod, BostonHousing[is.na(BostonHousing$rad), ])
ptratio_pred <- predict(anova_mod, BostonHousing[is.na(BostonHousing$ptratio), ])
