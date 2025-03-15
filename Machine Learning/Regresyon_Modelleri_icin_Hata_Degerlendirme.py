import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
data = {
    'experience_year': [5, 7, 3, 3, 2, 7, 3, 10, 6, 4, 8, 1, 1, 9, 1],
    'salary': [600, 900, 550, 500, 400, 950, 540, 1200, 900, 550, 1100, 460, 400, 1000, 380]
}


df = pd.DataFrame(data)

## 1-Verilen bias ve weight’e göre doğrusal regresyon model denklemini oluşturunuz.
#  Bias = 275, Weight= 90 (y’ =b+wx)

# y' = 275 + 90*experience_year

## 2-Oluşturduğunuz model denklemine göre tablodaki tüm deneyim yılları için maaş tahmini yapınız

def simple_model(dataframe, ey):
    df["salary_predict"] = 275 + 90 * ey

simple_model(df, df["experience_year"])

##  3-Modelin başarısını ölçmek için MSE, RMSE, MAE skorlarını hesaplayınız

# MSE
mse = mean_squared_error(df[["salary"]], df[["salary_predict"]])
# 4438.33

# RMSE
rmse = np.sqrt(mean_squared_error(df[["salary"]], df[["salary_predict"]]))
# 66.62

# MAE
mea = mean_absolute_error(df["salary"], df["salary_predict"])
# 54.33




