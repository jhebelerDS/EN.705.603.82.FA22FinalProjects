from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold

model_data = pd.read_csv('data.csv', index_col=0)
model_data = pd.concat([model_data['nodes'], model_data.drop(['nodes'], axis=1)], axis=1)

X = model_data.iloc[:, 1:]
y = model_data.iloc[:, 0]

# Implementing cross validation
# Instantiating the K-Fold cross validation object with 5 folds
k_folds = KFold(n_splits=5, shuffle=True)
# Iterating through each of the folds in K-Fold
for train_index, val_index in k_folds.split(X):
    print("Train :", train_index, "Test: ", val_index)
    X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Instantiating a RandomForestClassifier model
    rfc_model = XGBClassifier()

    # Fitting the X_train and y_train datasets to models below
    rfc_model.fit(X_train, y_train)

    # predictions for vaildation sets
    val_preds_xgb = rfc_model.predict(X_val)

    # Validation Metrics
    val_accuracy_xgb = accuracy_score(y_val, val_preds_xgb)
    val_confusion_matrix_xgb = confusion_matrix(y_val, val_preds_xgb)

    # Printing out the validation metrics
    print(f'Accuracy Score XGBClassifier: {val_accuracy_xgb}')
