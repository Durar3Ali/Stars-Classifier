import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# read the data & separate the x & y
df = pd.read_csv("Stars.csv")

X = df.drop("Type", axis=1)
y = df["Type"]

# encode the categorical columns (Color, Spectral_Class)
categorical_cols = ["Color", "Spectral_Class"]
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# scale the data
cols_to_scale = ["Temperature", "L", "R"]
X_to_scale = X[cols_to_scale]

scaler = StandardScaler()
X_scaled_part = scaler.fit_transform(X_to_scale)

X_scaled_part_df = pd.DataFrame(X_scaled_part, columns=cols_to_scale)
X_remaining = X.drop(columns=cols_to_scale)
X = pd.concat([X_scaled_part_df, X_remaining], axis=1) # the X final

#The Decision Tree Model
model = DecisionTreeClassifier(criterion="entropy", max_depth=3)
model.fit(X_train, y_train)

# evaluate the model
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy= ", round(acc*100, 2),'%')
print("Confusion Matrix=")
print(cm)
