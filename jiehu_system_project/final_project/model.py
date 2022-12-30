# This Python Script contains the different machine learning models applied to the breast
# cancer data

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def split_data(data):
    X = data.iloc[:, 1:-1]
    std = StandardScaler()
    X_std = std.fit_transform(X)
    X_std = pd.DataFrame(X_std, columns=X.columns)
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.30, random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test, X_std, y

def white_box_models(X_train, X_test, y_train, y_test):
    # Logistic Regression Model
    print("\n======================================================")
    print("Using white box models: ")
    lg_clf = LogisticRegression(random_state=0)
    lg_clf.fit(X_train, y_train)
    print("\nLogistic Regression Classifier accuracy score: ", accuracy_score(lg_clf.predict(X_test), y_test))
    print("Logistic Regression Coefficients:  ", lg_clf.coef_)

    # Decision Tree Model
    tree_clf = DecisionTreeClassifier(random_state=0)
    tree_clf.fit(X_train, y_train)
    print("\nDecision Tree Classifier accuracy score: ", accuracy_score(tree_clf.predict(X_test), y_test))

def black_box_models(X_train, X_test, y_train, y_test):
    print("\n============================================================")
    print("Using black box models:")

    # Random Forest Classifier
    forest_clf = RandomForestClassifier(criterion='entropy', random_state=0)
    forest_clf.fit(X_train, y_train)
    print("\nRandom Forest Classifier accuracy score: ", accuracy_score(forest_clf.predict(X_test), y_test))

    # SVM Classifier
    svm_clf = SVC(probability=True)
    svm_clf.fit(X_train, y_train)
    print("\nSVM Classifier accuracy score: ", accuracy_score(svm_clf.predict(X_test), y_test))

    # Neural Network Classifier
    nn_clf = MLPClassifier(max_iter=500)
    nn_clf.fit(X_train, y_train)
    print("\nNeural Network Classifier accuracy score: ", accuracy_score(nn_clf.predict(X_test), y_test))
    return [forest_clf, svm_clf, nn_clf]