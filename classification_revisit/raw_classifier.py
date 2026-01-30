from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd


# goal is to predict target label where 0 is malignant and 1 is benign 

data = load_breast_cancer()

X = data.data
y = data.target

x_df = pd.DataFrame(X)
print(x_df.describe())

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
#
