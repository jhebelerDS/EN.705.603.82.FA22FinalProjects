# This Python script is the entry point for the System Project

import pandas as pd
import numpy as np
from model import split_data, white_box_models, black_box_models
from explain_model import apply_PDP, apply_LIME, apply_SHAP
from IPython.display import display, HTML



def read_file(filename, column_names):
    # reads the input file
    data = pd.read_csv(filename)
    data.columns = column_names
    print(f"{filename} has been processed, and contain the following information:  ")
    print(data.head(5))
    print(f"\nThe dimensions of the data are: ", data.shape)
    print(data.info(verbose=True))
    return data

def data_preprocessing(data):
    # check values of categorical variable
    print(data["Bare Nuclei"].unique())
    data["Bare Nuclei"] = data["Bare Nuclei"].replace({'?': np.nan})

    # check and drop null values
    print(data.isnull().sum())
    data = data.dropna()
    data["Bare Nuclei"] = data["Bare Nuclei"].astype(int)
    return data

def main():
    file = "data/breast-cancer-wisconsin.data"
    column_names = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
        "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli",
        "Mitoses", "Class"]
    data = read_file(file, column_names)
    data = data_preprocessing(data)
    j = 5  # jth observation
    X_train, X_test, y_train, y_test, X_std, y = split_data(data)
    white_box_models(X_train, X_test, y_train, y_test)
    models = black_box_models(X_train, X_test, y_train, y_test)
    print("\n============================================================")
    print("Explain Models:")
    apply_PDP(models, X_std, column_names[1:-1])
    apply_LIME(models, X_train, X_test, j)
    apply_SHAP(models, X_test, j)
    print("Process has now completed.")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


