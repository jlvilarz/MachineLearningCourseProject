#We first load two packages that will be used later:
library("ggplot2");library("caret")
#We load the trining data set
dat <- read.csv("C:/Users/jlvilarz/Desktop/pml-training.csv")

#PREPROCESSING:
#We observe that the first column is just for numbering pruposes so 
#we eliminate it and show the data dimension
dat <- dat[,-1]
dim(dat)

#We notice that the $user_name column is of type character. As we will later do a correlation
#analysis we transform it to a numeric type so we can execute the fuunction cor():
summary(dat$user_name)
levels(dat$user_name) <- c(1:6)
levels(dat$user_name)
summary(dat$user_name)
dat$user_name <- as.numeric(dat$user_name)
summary(dat$user_name)

#Next we visualize the data set by means of summary() then notice that many of the columns are
#almost empty. For instance this is true for the following cases, among others:
summary(dat[,34]);summary(dat[,80]);summary(dat[,130]);

#We also remove the dates columns because we make the hypothesis that physical exercise achievement
# and dates are independent. So we remove all the time columns which are situates at the begining 
#of the data frame.In summary, we wash the data frame from the following colunms:
sec <- c(2:5,11:35,49:58,68:82,86:100,102:111,124:138,140:149)
sec
#This is the final raw data set to be used to build our model, together with its dimension:
dat.wash <- dat[,-sec]
dim(dat.wash)

#CROSS VALIDATION:
# We are going to consider three data sets:
#1-The TRAINING SET: extracted from the washed data frame "dat.wash" in a proportion of 2/3 of its total
#   number of cases (i.e. rows)
#2-The TESTING SET: This will be the other third of the "dat.wash" data frame.
#3-The VALIDATION SET: This is going to be the data frame loaded from the file "pml-testing.csv"

#PRE PROCESSING: 
#We will study if the training data set needs some pre-processing. By means of the correlation analysis
# the "training" data set we will conclude that it is worth executing a PCA analysis because that way we 
#simultaneously match the following goals:
#     a)Remove colinearities between data
#     b)Execute data centering-scaling.
#     c)Decrease the high dimensionality of the training data set
#consequently we will have to eccomplish exactly the same transformation to our testing and validation sets.

#Now we execute all this steps.

#Training and testing sets definition:
set.seed(123)
inTrain = createDataPartition(dat.wash$classe, p = 0.66)[[1]]
training = dat.wash[ inTrain,]
testing = dat.wash[-inTrain,]
dim(training)

#Correlation analysis: we fix a 0.8 of variance for 
#later decreasing of the pre-processed training data set.
M <- abs(cor(training[,-55]))
diag(M) <- 0
which(M>0.8,arr.ind=T)
#Thus we have detected many colinearitiies susceptible of being represented by means of
#Principal components analysis
#some graphical examples of those correlated variables are the following:
plot(training[,47],training[,48])
#Also:
plot(training[,12],training[,3])
#Or even:
plot(training[,21],training[,20])
#And so on...

#Therefore, we execute a PCA data pre-processing that will simultaneously 
#imply a centering and scaling of the data. We set the variance threshold equal to 0.8:
pre.proc <- preProcess(training[,-55],method="pca",thresh=.8)
pre.proc
#And we filter the training set through our PCA model:
trainPC <- predict(pre.proc,training[,-55])
dim(trainPC)
summary(trainPC)



#CHOICE OF THE PREDICTIVE MODEL:
#We fit a model by means of random forest. This is motivated by several considetration: 
#First of all we believe the problem is non linear as can be seen after plotting some of 
#the factors like for instance (in the following figure case-points are colored depending
#of their association to the training$classe factors):
featurePlot(training[,c("magnet_forearm_x","magnet_forearm_y","magnet_forearm_z")],training$classe,plot="pairs")
#Secondly, the covariate to be predicted, $classe, is a factor one, so it can be optimally 
#represented by the branches of a tree.
#Thirdly, the number of predictors is still too high (we have got 12 Principal Components
#substituting the original 55 covariates of the washed data) to try successive guesses on what
#should be the optimal set of predictors to be selected for our model

#thus we proceed applyin random forest methodology. We must warn the reader that the computations 
#have been so intensive that our notbook pentium core i5 with 8Gb of RAM has spent almost three
#hours to calculate the optimal fit. We embrace our calculation by a time clock to get a 
#precise idea of the time computing spent:
start.time <- Sys.time()
mod.fit1 <- train(training$classe ~ . ,method="rf",data=trainPC,prox=TRUE)
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

#Next we print the $finalModel output
print(mod.fit1$finalModel)

#and we plot also the successive etapes until we reached the final model, described byu means 
# of the Error variations form step to step in the random forest algorithm:
plot(mod.fit1$finalModel,main="Classification Tree")
#And we finally calculate the confusion matrix with its associated in-sample error (which by the 
#is null. This could be a matter of concern as it could denounce an overfitting situation.
confusionMatrix(training$classe,predict(mod.fit1,trainPC))



#Now we are going to spend the same treatment to aour testing set which is, remember,
#a third of the original washed pmal-training file: basically the same PCA analysis
testPC <- predict(pre.proc,testing[,-55])

#We apply the fitted model to the PCA testing set then calculate de confusion matrix, 
#to conclude that the out of sample error is more or less equl to 4%:
confusionMatrix(testing$classe,predict(mod.fit1,testPC))

#MODEL VALIDATION:
#We load the 20 test cases to validate the model
vali <- read.csv("C:/Users/jlvilarz/Desktop/pml-testing.csv")
#We are now going to give it the same treatment as the one given to the training set:
#We eliminate the first column as it is a counting one
vali <- vali[,-1]
dim(vali)
summary(vali)

#We convert the dat.test$user_name factor levels from character to numeric
summary(vali$user_name)
levels(vali$user_name) <- c(1:6)
levels(vali$user_name)
summary(vali$user_name)
vali$user_name <- as.numeric(vali$user_name)
summary(vali$user_name)


#We wash the data frame of all its empty or useless columns following the same pattern as for the training set:
sec <- c(2:5,11:35,49:58,68:82,86:100,102:111,124:138,140:149)
sec
vali.wash <- vali[,-sec]
dim(dat.wash)
#We finally summarize the validation data set:
summary(vali.wash)

#We watch the last column (nº55) in vali.wash to investigate its content. We conclude that this 
#should be removed has it seems to be  simply a case counter
vali.wash$problem_id

#We treat the vali.wash data set  by means of our previous PCA:
vali.washPC <- predict(pre.proc,vali.wash[,-55])
dim(vali.washPC)

#PREDICTIONS OF $CLASSE COVARIATE: 
#Our predictions are stores in the vector classe.pred:
classe.pred <- predict(mod.fit1,vali.washPC)
classe.pred
length(classe.pred)

#AND THAT'S ALL!!!
#THANK YOU FOR YOUR ATTENTION!!
