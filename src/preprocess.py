import pandas as pd
import numpy as np

# 1. 讀取資料 (請確保檔案路徑正確)
df = pd.read_csv('Titanic-Dataset.csv')

# 2. 特徵刪除 (手稿提到的 ID/Name delete)
# 刪除對預測生存無直接關聯的欄位
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# 3. 處理遺失值 (手稿中的 遺失值?)
# Age: 使用中位數填補
df['Age'] = df['Age'].fillna(df['Age'].median())

# Embarked: 使用眾數填補 (出現次數最多的港口)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 4. 類別轉換 (Encoding)
# 將性別 Sex 轉為數值 0 與 1
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

# 將登船港口 Embarked 進行 One-Hot Encoding
df = pd.get_dummies(df, columns=['Embarked'])

# 5. 查看前處理後的結果
print("前處理完成！目前資料欄位：", df.columns.tolist())
print(df.head())
