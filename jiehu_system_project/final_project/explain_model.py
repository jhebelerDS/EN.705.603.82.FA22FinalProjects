# This Python Script contains the implementation of several uniques to explain the blackbox models

from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import lime
from lime import lime_tabular
import numpy as np
import shap
from IPython.display import display, HTML


def apply_PDP(_models, _X, _features):
    # for each black box model, use Partial Dependence Plots to explain model predictions
    for model in _models:
        fig, ax = plt.subplots(figsize=(10, 10))
        PartialDependenceDisplay.from_estimator(model, _X, features=_features, ax=ax)
        ax.set_title(f"PDP for {model}")
        n = _models.index(model)
        plt.savefig(f'/output/pdp_{n}.png')
        plt.close()

def apply_LIME(_models, _Xtrain, _Xtest, n):
    # for each black box model, use LIME to explain model predictions
    for model in _models:
        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(_Xtrain), feature_names=_Xtrain.columns,
                                                           class_names=[2, 4], verbose=True, mode='classification')
        exp = explainer.explain_instance(_Xtest.iloc[n], model.predict_proba, num_features=9)
        exp.as_pyplot_figure()
        n = _models.index(model)
        plt.savefig(f'/output/LIME_{n}.png')
        plt.close()

def apply_SHAP(_models, _Xtest, n):
    # for each black box model, use SHAP to explain model predictions
    for model in _models:
        explainer = shap.Explainer(model.predict, _Xtest)
        shap_values = explainer(_Xtest)
        n = _models.index(model)
        shap.plots.bar(shap_values[n], show=False)
        plt.savefig(f'/output/SHAP_{n}.png')
        plt.close()
