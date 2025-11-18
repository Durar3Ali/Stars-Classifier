import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

# read the data &separate the x & y
df = pd.read_csv("Stars.csv")

X = df.drop("Type", axis=1)
y = df["Type"]

# encode the categorical columns (Color, Spectral_Class)
categorical_cols = ["Color", "Spectral_Class"]
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# split the data 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# build the logistic regression model
model = LogisticRegression(max_iter=2000, multi_class="multinomial")

param_grid = {"C": [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(
  estimator=model, #estimator = model
  param_grid=param_grid,
  cv=5,
  scoring="accuracy",
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# evaluate the model
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy=", round(acc*100, 2),'%')
print("Confusion Matrix=")
print(cm)