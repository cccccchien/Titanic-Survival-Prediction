import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 讀取你上傳到 GitHub 的資料
df = pd.read_csv('Titanic-Dataset.csv')

# 設定繪圖風格
sns.set_theme(style="whitegrid")

# 建立一個畫布，準備放兩張圖
plt.figure(figsize=(12, 5))

# 圖一：性別 vs 生存人數 (驗證手稿中的 Sex 特徵)
plt.subplot(1, 2, 1)
sns.countplot(data=df, x='Sex', hue='Survived', palette='viridis')
plt.title('Survival Count by Sex')
plt.xlabel('Sex (0=Female, 1=Male)')

# 圖二：船票等級 vs 生存率 (驗證手稿中的 Pclass 特徵)
plt.subplot(1, 2, 2)
sns.barplot(data=df, x='Pclass', y='Survived', ci=None, palette='magma')
plt.title('Survival Rate by Pclass')
plt.ylabel('Survival Probability')

plt.tight_layout()
plt.show()
