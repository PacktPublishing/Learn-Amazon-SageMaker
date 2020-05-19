install.packages("rjson")
library("rjson")

hp <- fromJSON(file = '/opt/ml/input/config/hyperparameters.json')
print(hp)
normalize <- hp$normalize

data <- read.table(file = '/opt/ml/input/data/training/housing.csv', header=T)
if (normalize) {
    data <- as.data.frame(scale(data))
}
summary(data)

model = lm(medv~.,data)
summary(model)

saveRDS(model, '/opt/ml/model/model.rds')