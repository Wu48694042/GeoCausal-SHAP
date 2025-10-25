# GeoCausal-SHAP
GeoCausal SHAP's frame work.

It has mode 1 and mode 2, two different complex SHAP calculations designed by me. It also has three models, random forest, XGBoost, and MLP, to make the prediction. 

1.The main workflow is in the GeoCausal SHAP.py, where it will use the machine learning method to make a prediction first from the methods integrated in utils.py. 

2.And causal discovery algorithms are integrated in causal_discovery.py, which provides a causal DAG for supporting further SHAP calculation.

3.Then do the SHAP calculations using algorithms in SHAP_explainer.py. 

4.The setting of machine learning methods for prediction is in the utils.py.
