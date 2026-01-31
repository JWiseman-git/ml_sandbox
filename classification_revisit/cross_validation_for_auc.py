import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])

data = datasets.load_breast_cancer()
X = data.data
y = data.target

# data classification task is binary - used logistic regression
# k fold partitions the dataset into k subset so test/train data

cross_val = KFold(n_splits=6, random_state=42, shuffle=True)

# update after iterations over folds

tprs, aucs = [], []
mean_fpr = np.linspace(0, 1, 1000)

fig, ax = plt.subplots()
for index, (train, test) in enumerate(cross_val.split(X, y)):
    pipeline.fit(X[train], y[train])
    plot = RocCurveDisplay.from_estimator(
        pipeline, X[test], y[test],
        name="ROC fold {}".format(index),
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, plot.fpr, plot.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(plot.roc_auc)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic with CV",
)
plt.savefig("roc_cv.jpeg")