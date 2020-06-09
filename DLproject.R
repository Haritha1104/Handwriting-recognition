
#Haritha Ramachandran


library(keras)
library(dplyr)
library(tensorflow)
load("data_usps_digits.RData")
library(reticulate)

tensorflow::tf$random$set_seed(1)


#data preparation
y_train <-to_categorical(y_train,10)
y_test <- to_categorical(y_test ,10) 


range_norm <-function(x,a =0,b =1) {
  ( (x-min(x))/(max(x)-min(x)) )*(b-a)+a 
}


x_train <-apply(x_train,2, range_norm)
range(x_train)   #check whether range is 0,1
x_test <-apply(x_test,2, range_norm)
range(x_test)
x_train<- as.matrix(x_train)
x_test <- as.matrix(x_test)

val <-sample(1:nrow(x_test),1003)# there are 2007 images in x_test
test <-setdiff(1:nrow(x_test), val)
x_val <-x_test[val,]
y_val <-y_test[val,]
x_test <-x_test[test,]
y_test <-y_test[test,]

library(tfruns)


# run ---------------------------------------------------------------



layer1_nodes = c(256,128)
dropout1_rates=c(0, 0.3, 0.4, 0.5)
layer2_nodes=c(128,64)
dropout2_rates=c(0, 0.3, 0.4, 0.5)


runs <- tuning_run("A3_conf.R",
                   runs_dir = "runs_assignment", 
                   flags = list(
                     dense_units1 =c(256,128),
                     dropout1 = c(0, 0.3, 0.4, 0.5),
                     dense_units2 =c(128,64),
                     dropout2 = c(0, 0.3, 0.4, 0.5)), 
                   sample = 0.2)






