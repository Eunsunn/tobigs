setwd("C:/Users/dmstj/Desktop/regression")

data = read.csv("Auction_master_train.csv", fileEncoding = "utf-8")

str(data)

#결측치 많은 열 삭제 : addr_bunji2, Specific, road_bunji2,addr_li
colSums(is.na(data))

data = subset(data, select = -c(addr_bunji2, Specific, road_bunji2,addr_li))

#독립변수와 종속변수의 plot
plot(data$Appraisal_company, data$Hammer_price) #Appraisal_company에따라 차이가 크지 않음
plot(data$Auction_key, data$Hammer_price)
plot(data$Creditor, data$Hammer_price)
plot(data$Apartment_usage, data$Hammer_price)

# 그 외 addr_si, addr_dong,addr_san,addr_bunji1,addr_etc, road_name,road_bunji1 등 주소관련 칼럼 삭제
# : factor의 요인들이 너무 많음, 좀더 유의미한 addr_do 변수와 위도 경도인 point_x, point_y 사용
# Final_result와 Close_result는 모든 데이터가 같은 값을 가지므로 회귀분석시 삭제
# Apartment_usage, Appraisal_company, Creditor 등 plot을 그렸을때 요인별 차이가 크지 않으므로 삭ㅈ
data = subset(data, select = -c(Appraisal_company, Creditor, Auction_key, addr_si,
                                addr_dong, addr_san, addr_bunji1, addr_etc, road_name,road_bunji1,
                                Final_result, Close_result, Apartment_usage))


#시간 데이터를 시계열 자료로 바꾸기
data$Appraisal_date = as.Date(data$Appraisal_date)
data$First_auction_date = as.Date(data$First_auction_date)
data$Final_auction_date = as.Date(data$Final_auction_date)
data$Close_date = as.Date(data$Close_date)
data$Preserve_regist_date = as.Date(data$Preserve_regist_date)

#파생변수 만들기 >>2개더 생각해보기
data$date = data$Final_auction_date - data$First_auction_date
data$price_diff = data$Total_appraisal_price - data$Minimum_sales_price
data$location = data$point.x + data$point.y

#다중공선성이 커지므로 파생변수 생성에 이용한 변수들을 삭제
data = subset(data, selec = -c(Final_auction_date, First_auction_date, Total_appraisal_price, Minimum_sales_price, point.x, point.y))  
  
#데이터 train, test 분리 7:3
#install.packages("caret")
library(caret)
idx <- createDataPartition(y = data$Hammer_price, p = 0.7, list =FALSE)
#7:3으로 나눠라
train<- data[idx,]
test <- data[-idx,]

str(train)

lm = lm(Hammer_price ~ ., data = train)
summary(lm)

#회귀결과 Bid_class, Auction_count, Auction_miscarriage_count, price_diff의 회귀계수 p값만 유의하므로
#나머지 변수들을 제거하는게 바람직하다.

# cross validation
library(caret)
cv <- trainControl(method = "cv", number = 5, verbose = T) #5개로 교차검증 

# lm to test set
train.lm <- train(Hammer_price~.,train, method = "lm", trControl =cv) #method가 회귀
predict.lm <- predict(train.lm,test[,-19])
y <- test[,19]
RMSE(predict.lm,y) 

#모든 독립변수와 종속변수의 관계 시각화 
plot(Hammer_price~.,data = data)
plot(lm)

#특별히 회귀식에서 회귀계수가 유의미한 (회귀관계로 종속변수를 설명할 수 있는) 변수의 plot
plot(data$Bid_class, data$Hammer_price)
plot(data$Auction_count, data$Hammer_price)
plot(data$Auction_miscarriage_count, data$Hammer_price)
plot(data$price_diff, data$Hammer_price)
