###################################Processing Script for Linear Regression##########################################
install.packages("RMySQL")
install.packages("ggplot2")
install.packages("Rserve")
install.packages("sqldf")
install.packages("caret")

library(RMySQL)
library(ggplot2)
library(Rserve)
library(sqldf)
library(caret)


mydb = dbConnect(MySQL(), user='XXXXX', password='XXXXXXX', dbname='XXXXXXXX', host='XX.XXX.XXX.XXX')

tb_testset_info_data <- as.data.frame(dbReadTable(mydb,'tb_testset_info'))
colnames(tb_testset_info_data)
nrow(tb_testset_info_data)
write.csv(tb_testset_info_data,file = "tb_testset_info_data_26may.csv", row.names = FALSE )

tb_testset_execution_info_data <- as.data.frame(dbReadTable(mydb,'tb_testset_execution_info'))
colnames(tb_testset_execution_info_data)
nrow(tb_testset_execution_info_data)
write.csv(tb_testset_execution_info_data,file = "tb_testset_execution_info_data_26may.csv", row.names = FALSE )

tb_testcase_info_data <- as.data.frame(dbReadTable(mydb,'tb_testcase_info'))
colnames(tb_testcase_info_data)
nrow(tb_testcase_info_data)
write.csv(tb_testcase_info_data,file = "tb_testcase_info_data_26may.csv", row.names = FALSE )

tb_testcase_execution_info_data <- as.data.frame(dbReadTable(mydb,'tb_testcase_execution_info'))
colnames(tb_testcase_execution_info_data)
nrow(tb_testcase_execution_info_data)
write.csv(tb_testcase_execution_info_data,file = "tb_testcase_execution_info_data_26may.csv", row.names = FALSE )

tb_script_info_data <- as.data.frame(dbReadTable(mydb,'tb_script_info'))
colnames(tb_script_info_data)
nrow(tb_script_info_data)
write.csv(tb_script_info_data,file = "tb_script_info_data_26may.csv", row.names = FALSE )

tb_command_results_data <- as.data.frame(dbReadTable(mydb,'tb_command_results'))
colnames(tb_command_results_data)
nrow(tb_command_results_data)
write.csv(tb_command_results_data,file = "tb_command_results_data_26may.csv", row.names = FALSE )

tb_object_info_data <- as.data.frame(dbReadTable(mydb,'tb_object_info'))
colnames(tb_object_info_data)
nrow(tb_object_info_data)
write.csv(tb_object_info_data,file = "tb_object_info_data_26may.csv", row.names = FALSE )

tb_action_info_data <- as.data.frame(dbReadTable(mydb,'tb_action_info'))
colnames(tb_action_info_data)
nrow(tb_action_info_data)
write.csv(tb_action_info_data,file = "tb_action_info_data_26may.csv", row.names = FALSE )

tb_protocol_type_info_data <- as.data.frame(dbReadTable(mydb,'tb_protocol_type_info'))
colnames(tb_protocol_type_info_data)
nrow(tb_protocol_type_info_data)
write.csv(tb_protocol_type_info_data,file = "tb_protocol_type_info_data_26may.csv", row.names = FALSE )

tb_mel_config_basic_data <- as.data.frame(dbReadTable(mydb,'tb_mel_config_basic'))
colnames(tb_mel_config_basic_data)
nrow(tb_mel_config_basic_data)
write.csv(tb_mel_config_basic_data,file = "tb_mel_config_basic_data_26may.csv", row.names = FALSE )

#tb_testcase_execution_info_data <- tb_testcase_execution_info_data[!is.na(tb_testcase_execution_info_data$end_time),]
#st <- strptime(tb_testcase_execution_info_data$start_time, "%Y-%m-%d %H:%M:%S")
#et <- strptime(tb_testcase_execution_info_data$end_time, "%Y-%m-%d %H:%M:%S")
#for(i in 1:nrow(tb_testcase_execution_info_data))
#{
#  tb_testcase_execution_info_data$time_diff[i]<- as.vector(difftime(tb_testcase_execution_info_data$end_time[i],tb_testcase_execution_info_data$start_time[i],unit="mins"))
#}
#colnames(tb_testcase_execution_info_data)
#head(tb_testcase_execution_info_data)

#Getting required columns from testset and execution table
testset_data <- merge(tb_testset_execution_info_data[,c("id_testset_run","testset_id","user_name","program_id","team_name")],tb_testset_info_data[,c("id_testset","testset_name")], by.x = c("testset_id"), by.y = c("id_testset"))
colnames(testset_data)
nrow(testset_data)
write.csv(testset_data,file = "testset_data26may.csv", row.names = FALSE )

#Getting required columns from mel_config_basic and testset table created above to get data 
#testset_data <- merge(testset_data, tb_mel_config_basic_data[,c("testset_execution_id","model","platform","revision")], by.x = c("testset_id"), by.y = c("testset_execution_id"))
#testset_data <- merge(testset_data, tb_mel_config_basic_data[,c("testset_execution_id","model","platform")], by.x = c("testset_id"), by.y = c("testset_execution_id"))
testset_data <- merge(testset_data, tb_mel_config_basic_data[,c("testset_execution_id","model","platform")], by.x = c("id_testset_run"), by.y = c("testset_execution_id"))
colnames(testset_data)
nrow(testset_data)
656-9906
testset_data <- testset_data[!duplicated(testset_data$id_testset_run),]
write.csv(testset_data,file = "testset_data26may.csv", row.names = FALSE )

#Getting required columns from testcase and execution table
testcase_data <- merge(tb_testcase_execution_info_data[,c("id_testcase_run","testset_id","testcase_id","script_id","start_time","end_time","status","testcase_runtime_name")],tb_testcase_info_data[,c("id_testcase","testcase_name")], by.x = c("testcase_id"), by.y = c("id_testcase"))
colnames(testcase_data)
nrow(testcase_data)

#Getting required columns from above obtained testcase and script table
testcase_data <- merge(testcase_data,tb_script_info_data[,c("id_script","script_name")], by.x = c("script_id"), by.y = c("id_script"))
colnames(testcase_data)
nrow(testcase_data)
write.csv(testcase_data,file = "testcase_data26may.csv", row.names = FALSE )

current_date <- format(Sys.Date(),format = "%Y-%m-%d %H:%M:%S")
current_date

rm(merge_data1)
#testset_data$testset_id <- NULL
#Merging data from testset_execution and testcase_execution table obtained above
merge_data1 <- merge(testcase_data, testset_data[,c("id_testset_run","user_name","program_id","team_name","testset_name","model","platform")], by.x = c("testset_id"), by.y = c("id_testset_run"))
colnames(merge_data1)
nrow(merge_data1)
write.csv(merge_data1,file = "merge_data1_26may.csv", row.names = FALSE )
merge_data1 <- read.csv("merge_data1_26may.csv", header = TRUE)

merge_data1 <- merge_data1[!is.na(merge_data1$end_time),]
#Using parallel computing
install.packages("parallel")
install.packages("foreach")
install.packages("doParallel")
install.packages("quantmod")
library(parallel)
library(doParallel)
library(foreach)
library(quantmod)

chartSeries(seq(1:100))
core_num <- detectCores()-4
core_num
chunk_start_pos <- 1
chunk_end_pos <- chunk_start_pos+core_num-1

colnames(merge_data1)
cl <- makeCluster(core_num)
cl
clusterExport(cl,"merge_data1")

parLapply(cl,1:nrow(merge_data1), function(x) {merge_data1$time_diff<- difftime(merge_data1$end_time,merge_data1$start_time,unit="mins")})
merge_data1$time_diff<- foreach(i=1:nrow(merge_data1)) %dopar% difftime(merge_data1$end_time,merge_data1$start_time,unit="mins")
merge_data1 <- merge_data1[!is.na(merge_data1$end_time),]

stopCluster(c1)

test_merge_data1 <- merge_data1[1:100,]
test_merge_data1$temp_time_diff <- foreach(j=1:nrow(test_merge_data1), .combine = 'rbind' ) %dopar% 
{
  difftime(test_merge_data1$end_time[j],test_merge_data1$start_time[j],unit="mins")
}
print(proc.time())
test_merge_data1
temp_time_diff
write.csv(test_merge_data1$temp_time_diff,file = "temp_time_diff.csv", row.names = FALSE )
write.csv(test_merge_data1,file = "test_merge_data1.csv", row.names = FALSE )

#Fetching datarows based on prior months
install.packages("lubridate")
library(lubridate)
month(merge_data1$end_time)
year(merge_data1$end_time)

for(i in 1:nrow(test_merge_data1))
{
  #trainset <- trsf[which(trsf$end_time[i] < as.Date(current_date)),]
  trainset <- test_merge_data1[which(test_merge_data1$end_time[i] < as.Date("2016-01-07 00:00:00")),]
  #testset <- trsf[which(trsf$end_time[i] > current_date),]
  testset <- test_merge_data1[which(test_merge_data1$end_time[i] > as.Date("2016-01-07 00:00:00")),]
}

write.csv(trainset,file = "trainset_26may.csv", row.names = FALSE )
write.csv(testset,file = "testset_26may.csv", row.names = FALSE )


#using Big Data for above for loop working
install.packages("bigmemory")
install.packages("biganalytics")
install.packages("bigtabulate")
install.packages("biglm")
library(bigmemory)
library(biganalytics)
library(bigtabulate)
library(biglm)

write.csv(merge_data1,file = "merge_data1_demo.csv", row.names = FALSE )

getwd()





#Calculating the time_diff from start_time and end_time within testcase_data
merge_data1 <- merge_data1[!is.na(merge_data1$end_time),]
merge_data1 <- merge_data1[!(is.na(merge_data1$model) | merge_data1$model == ""),]
merge_data1 <- merge_data1[!(is.na(merge_data1$platform) | merge_data1$platform == ""),]
#st <- strptime(tb_testcase_execution_info_data$start_time, "%Y-%m-%d %H:%M:%S")
#et <- strptime(tb_testcase_execution_info_data$end_time, "%Y-%m-%d %H:%M:%S")
for(i in 1:nrow(merge_data1))
{
  merge_data1$end_date[i]<- format(as.Date(merge_data1$end_time[i], format = "%Y-%m-%d  %H:%M:%S","%Y-%m-%d"))
  merge_data1$time_diff[i]<- as.vector(difftime(merge_data1$end_time[i],merge_data1$start_time[i],unit="mins"))
}
colnames(merge_data1)
head(merge_data1)
write.csv(merge_data1,file = "merge_data1_26may_try.csv", row.names = FALSE )
merge_data1 <- read.csv("merge_data1_26may_try.csv", header = TRUE)


temp_data <- as.data.frame(merge_data1[1:1000,])
#temp_data <- temp_data[,-which(names(temp_data) %in% c("testset_id","script_id","testcase_id","id_testcase_run","start_time","end_time","testcase_runtime_name","program_id","platform","end_date"))]
temp_data <- temp_data[,-which(names(temp_data) %in% c("testset_id","script_id","testcase_id","id_testcase_run","start_time","end_time","testcase_runtime_name","program_id","platform","end_date"))]
colnames(temp_data)
predictors <- trainset[,-which(names(trainset) %in% c("start_time","end_time","end_date","time_diff"))]
#merge_data1_categ <- merge_data1[,c(1:4,7:16)]
merge_data1_categ <- temp_data[,c(1:7)]
#merge_data1_cont <- merge_data1[,c(5:6,17:18)]
merge_data1_cont <- as.numeric(temp_data[,c(8)])
names(merge_data1_cont)[names(merge_data1_cont) == 'merge_data1_cont'] <- "time_diff"
colnames(merge_data1_cont)
#Reordering columns to keep time_diff at end
#merge_data1 <- merge_data1[,c(1:7,9:17,8)]
colnames(merge_data1_categ)
rm(trsf)
gc()
###Dummy only categorical variables except start time and end time
dmy <- dummyVars("~ .", data = merge_data1_categ)
trsf <- data.frame(predict(dmy, newdata = merge_data1_categ))
colnames(trsf)
ncol(trsf)
nrow(trsf)
head(trsf)

##New way of parsing categorical variables
n <- nrow(merge_data1_categ)
nlevels <- sapply(merge_data1_categ, nlevels)
i <- rep(seq_len(n), ncol(merge_data1_categ))
j <- unlist(lapply(merge_data1_categ, as.integer)) +
  rep(cumsum(c(0, head(nlevels, -1))), each = n)
x <- 1
library(Matrix)
y <- sparseMatrix(i = i, j = j, x = x)
nlevels
library(MASS)
write.matrix(y,"matrix.txt")

trsf <- as.data.frame(cbind(trsf,merge_data1_cont))
names(trsf)[names(trsf) == 'merge_data1_cont'] <- "time_diff"
write.csv(trsf,file = "trsf_new_1000_rows_1June16.csv", row.names = FALSE )
trsf$end_date <- as.Date(as.character(trsf$end_date), "%Y-%m-%d")
trsf$end_time <- as.Date(as.character(trsf$end_time), "%Y-%m-%d")
trainset <- trsf[which(trsf$end_date < as.Date(current_date)),]
testset <- trsf[which(trsf$end_date >= as.Date(current_date)),]

write.csv(trainset,file = "trainset.csv", row.names = FALSE )
write.csv(testset,file = "testset.csv", row.names = FALSE )


for(i in 1:nrow(trsf))
{
  #trainset <- trsf[which(trsf$end_time[i] < as.Date(current_date)),]
  trainset <- trsf[which(trsf$end_date[i] < as.Date("2016-01-20")),]
  #testset <- trsf[which(trsf$end_time[i] > current_date),]
  testset <- trsf[which(trsf$end_date[i] > as.Date("2016-01-20")),]
}


#Reordering columns to keep time_diff at end
merge_data1 <- merge_data1[,c(1:6,8:15,7)]

#Merging data from obtained merge testcase_execution data above and mel_config_basic table
#merge_data1 <- merge(merge_data1, tb_mel_config_basic_data[,c("testset_execution_id","model","platform","revision")], by.x = c("testset_id"), by.y = c("testset_execution_id"))
#colnames(merge_data1)
#nrow(merge_data1)

write.csv(merge_data1,file = "merge_data1_demo.csv", row.names = FALSE )
write.csv(merge_data1,file = "merge_data1_26May.csv", row.names = FALSE )

#merge_data1 <- merge_data1[!is.na(merge_data1$id_testset_run),]
merge_data1 <- merge_data1[!is.na(merge_data1$testset_id),]
merge_data1 <- merge_data1[!is.na(merge_data1$testcase_id),]
merge_data1 <- merge_data1[!is.na(merge_data1$id_testcase_run),]
merge_data1 <- merge_data1[!is.na(merge_data1$script_id),]
merge_data1 <- merge_data1[!is.na(merge_data1$status),]
merge_data1 <- merge_data1[!is.na(merge_data1$testcase_runtime_name),]
merge_data1 <- merge_data1[!is.na(merge_data1$testcase_name),]
merge_data1 <- merge_data1[which(!merge_data1$user_name == "unknown"),]
#merge_data1 <- merge_data1[!is.na(merge_data1$program_id),]
merge_data1 <- merge_data1[!(is.na(merge_data1$team_name) | merge_data1$team_name =="Not Set" | merge_data1$team_name =="unknown"),]
merge_data1 <- merge_data1[!is.na(merge_data1$testset_name),]
merge_data1 <- merge_data1[!(is.na(merge_data1$model) | merge_data1$model == ""),]
merge_data1 <- merge_data1[!(is.na(merge_data1$platform) | merge_data1$platform == ""),]
merge_data1 <- merge_data1[!(is.na(merge_data1$program_id) | merge_data1$program_id == "unknown"),]
write.csv(merge_data1,file = "merge_data1_demo.csv", row.names = FALSE )
#merge_data1 <- merge_data1[!(is.na(merge_data1$revision) | merge_data1$revision == "unknown"),]
#merge_data1$framework_id <- NULL

detach(merge_data1)
attach(merge_data1)

dmy <- dummyVars("~.", data = merge_data1)
#dmy <- dummyVars("~.", data = new_data)
#trsf <- data.frame(predict(dmy, newdata = new_data))
trsf <- data.frame(predict(dmy, newdata = merge_data1))
colnames(trsf)
head(trsf)
ncol(trsf)
set.seed(1234567890)

detach(trsf)
attach(trsf)

## 75% of the sample size
smp_size <- floor(0.75 * nrow(trsf))

## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(trsf)), size = smp_size)

trainset <- trsf[train_ind, ]
nrow(trainset)
ncol(trainset)
head(trainset)
testset <- trsf[-train_ind, ]
nrow(testset)

colnames(trainset)
View(trainset[,c("time_diff")])

#predictors <- trainset[,-which(names(trainset) == "time_diff")]
#predictors <- trainset[,-which(names(trainset) %in% c("start_time","end_time","end_date","time_diff"))]
predictors <- trainset[,-which(names(trainset) %in% c("time_diff"))]
colnames(predictors)
ncol(predictors)
rm(predictors)
rm(parm)
#predictors <- as.vector(colnames(trainset[,-which(names(trainset) == "time_diff")]))
#predictors <- as.vector(colnames(trainset[,-which(names(trainset) %in% c("start_time","end_time","end_date","time_diff"))]))
#predictors <- as.vector(colnames(trainset[,-which(names(trainset) %in% c("start_time","end_time","time_diff"))]))
predictors <- as.vector(colnames(trainset[,-which(names(trainset) %in% c("time_diff"))]))
predictors <- as.vector(colnames(trainset[,-which(names(trainset) %in% c("merge_data1_cont"))]))
head(predictors)
predictors
parm <- as.formula(paste('time_diff ~', paste(predictors,collapse = '+')))
parm

library(biganalytics)
blm  <- biglm.big.matrix (parm,data=trainset)
blm  <- biglm(parm,data=trainset)
summary(blm)
deviance(blm)
AIC(blm)
tail(summary(blm))
blm
#Running Linear Regression
linear_model <- lm(parm)
summary(linear_model)
#plot(linear_model)

#predictions <- predict.lm(linear_model, testset)
predictions <- predict.biglm(blm, newdata = testset, type = c("link"))

testset<- cbind(testset,predictions, se.fit = TRUE)
testset

for(i in 1:nrow(testset)){
  testset$lower_bound[i] <- 0.9*testset$predictions[i]
  testset$upper_bound[i] <- 1.1*testset$predictions[i]
  #if(testset$lower_bound[i]<=testset$time_diff[i] & testset$time_diff[i]<=testset$upper_bound[i]){
  if(testset$time_diff[i]<=testset$upper_bound[i]){
    testset$flag[i]<- 0 }
  else {
    testset$flag[i]<- 1 }
}
colnames(testset)

#write.csv(testset,file = "testset.csv", row.names = FALSE )
write.csv(testset,file = "testset_new_1June16.csv", row.names = FALSE )


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
##Validating the Test Data set for prediction values
getwd()
test_data_valid <- read.csv("testset_prediction_10June.csv", header = TRUE)
test_data_valid_sample <- test_data_valid[sample(nrow(test_data_valid),25),]
write.csv(test_data_valid_sample,file = "test_data_valid_sample_10June16.csv", row.names = FALSE )


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
##Calculating the mean based on individual level of attributes

colnames(merge_data1)
merge_data1 <- merge_data1[,-which(names(merge_data1) %in% c("testset_id","script_id","testcase_id","id_testcase_run","start_time","end_time","testcase_runtime_name","program_id","platform","end_date"))]
colnames(merge_data1)
temp_data <- read.table(merge_data1[1:1000,],header = TRUE)
head(temp_data,20)
detach()
attach(temp_data)
aggregate(as.matrix(temp_data[,8],as.list(temp_data[,1:7]), FUN = mean))
aggregate(as.matrix(temp_data$time_diff,as.list(),mean))
install.packages("data.table")
library(data.table)
temp_data[,lapply(.SD,mean),by=temp_data[1:7]]
colnames(temp_data)
nrow(temp_data)
##WORKING CODE BELOW
temp_data <- data.table(temp_data) 
temp_data_mean <- temp_data[,Mean:=mean(time_diff),by=list(status,testcase_name,script_name,user_name,team_name,testset_name,model)]
temp_data_median <- temp_data[,Median:=median(time_diff),by=list(status,testcase_name,script_name,user_name,team_name,testset_name,model)]
write.csv(temp_data_mean,file = "test_data_mean_2June16.csv", row.names = FALSE )
write.csv(temp_data_median,file = "test_data_median_2June16.csv", row.names = FALSE )
getwd()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#ALTERNATE APPROACH USING MEDIAN TO FIND ABNORMAL TEST CASE EXECUTIONS


mydb = dbConnect(MySQL(), user='XXXXX', password='XXXXXXX', dbname='XXXXXXXX', host='XX.XXX.XXX.XXX')

tb_testset_info_data <- as.data.frame(dbReadTable(mydb,'tb_testset_info'))

tb_testset_execution_info_data <- as.data.frame(dbReadTable(mydb,'tb_testset_execution_info'))

tb_testcase_info_data <- as.data.frame(dbReadTable(mydb,'tb_testcase_info'))

tb_testcase_execution_info_data <- as.data.frame(dbReadTable(mydb,'tb_testcase_execution_info'))

tb_script_info_data <- as.data.frame(dbReadTable(mydb,'tb_script_info'))

#tb_command_results_data <- as.data.frame(dbReadTable(mydb,'tb_command_results'))

tb_object_info_data <- as.data.frame(dbReadTable(mydb,'tb_object_info'))

tb_action_info_data <- as.data.frame(dbReadTable(mydb,'tb_action_info'))

tb_protocol_type_info_data <- as.data.frame(dbReadTable(mydb,'tb_protocol_type_info'))

tb_mel_config_basic_data <- as.data.frame(dbReadTable(mydb,'tb_mel_config_basic'))

testset_data <- merge(tb_testset_execution_info_data[,c("id_testset_run","testset_id","user_name","program_id","team_name")],tb_testset_info_data[,c("id_testset","testset_name")], by.x = c("testset_id"), by.y = c("id_testset"))
colnames(testset_data)
nrow(testset_data)

testset_data <- merge(testset_data, tb_mel_config_basic_data[,c("testset_execution_id","model","platform")], by.x = c("id_testset_run"), by.y = c("testset_execution_id"))
colnames(testset_data)
nrow(testset_data)
testset_data <- testset_data[!duplicated(testset_data$id_testset_run),]


testcase_data <- merge(tb_testcase_execution_info_data[,c("id_testcase_run","testset_id","testcase_id","script_id","start_time","end_time","status","testcase_runtime_name")],tb_testcase_info_data[,c("id_testcase","testcase_name")], by.x = c("testcase_id"), by.y = c("id_testcase"))
colnames(testcase_data)
nrow(testcase_data)

testcase_data <- merge(testcase_data,tb_script_info_data[,c("id_script","script_name")], by.x = c("script_id"), by.y = c("id_script"))
colnames(testcase_data)
nrow(testcase_data)

merge_data1 <- merge(testcase_data, testset_data[,c("id_testset_run","user_name","program_id","team_name","testset_name","model","platform")], by.x = c("testset_id"), by.y = c("id_testset_run"))
colnames(merge_data1)
nrow(merge_data1)

merge_data1 <- merge_data1[!is.na(merge_data1$end_time),]
merge_data1 <- merge_data1[!(is.na(merge_data1$model) | merge_data1$model == ""),]
merge_data1 <- merge_data1[!(is.na(merge_data1$platform) | merge_data1$platform == ""),]

for(i in 1:nrow(merge_data1))
{
  merge_data1$duration[i]<- as.vector(difftime(merge_data1$end_time[i],merge_data1$start_time[i],unit="mins"))
}
colnames(merge_data1)
write.csv(merge_data1,file = "merge_data1_20June.csv", row.names = FALSE )
merge_data1 <- read.csv("merge_data1_20June.csv", header = TRUE)

temp_data <- as.data.frame(merge_data1)
temp_data <- temp_data[,-which(names(temp_data) %in% c("testset_id","script_id","testcase_id","id_testcase_run","end_time","testcase_runtime_name","program_id","platform"))]
colnames(temp_data)

install.packages("data.table")
library(data.table)
temp_data <- data.table(temp_data) 
temp_data_mean <- temp_data[,Mean:=mean(duration),by=list(status,testcase_name,script_name,user_name,team_name,testset_name,model)]
temp_data_mean_median <- temp_data_mean[,Median:=median(duration),by=list(status,testcase_name,script_name,user_name,team_name,testset_name,model)]
#write.csv(temp_data_mean,file = "test_data_mean_20June16.csv", row.names = FALSE )
write.csv(temp_data_mean_median,file = "test_data_median_20June16.csv", row.names = FALSE )
getwd()
temp_data_mean_median <- read.csv("test_data_median_20June16.csv", header = TRUE)

for(i in 1:nrow(temp_data_mean_median)){
  temp_data_mean_median$upper_bound[i] <- 1.1*temp_data_mean_median$Median[i]
  if(temp_data_mean_median$duration[i]<=temp_data_mean_median$upper_bound[i]){
    temp_data_mean_median$flag[i]<- 0 }
  else {
    temp_data_mean_median$flag[i]<- 1 }
}
colnames(temp_data_mean_median)

write.csv(temp_data_mean_median,file = "test_data_median_flag_20June16.csv", row.names = FALSE )
