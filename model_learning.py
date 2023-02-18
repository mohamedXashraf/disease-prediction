import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# --------------------------------------- Model Definition --------------------------------------- #
model = DecisionTreeClassifier()

# --------------------------------------- Model Training --------------------------------------- #
train_data = pd.read_csv("dataset/dataset_processed.csv")
X, y = train_data.drop("Disease", axis=1), train_data['Disease']
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3)

# Train
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_cv)

# Evaluation
print(classification_report(y_cv, predictions))
