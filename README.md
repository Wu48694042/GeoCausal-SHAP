# GeoCausal-SHAP
Dataset and algorithms used for my experiments of my master thesis related to SHAP calculations: GeoCausal SHAP - A Geovisual Analytics Framework for Explaining AI Combining SHAP and Causal Discovery

When testing GeoCausal SHAP, using the dataset I provide in the dataset, and altogether four .py files (GeoCausal SHAP.py, Causal_discovery.py, shap_explainer.py, utils.py) at the same time, datapreprocessing methods for datasets are all in the GeoCausal SHAP.py, you can choose different datasets for testing in the load dataset part. Changing different causal discovery mode, shap calculation mode, and predition model in the config as well.

Environment file is also provided.

GeoCausal SHAP's framework:

It has mode 1 and mode 2, two different complex SHAP calculations designed by me. It also has three models, random forest, XGBoost, and MLP, to make the prediction. 

1.The main workflow is in the GeoCausal SHAP.py, where it will use the machine learning method to make a prediction first from the methods integrated in utils.py. 

2.And causal discovery algorithms are integrated in causal_discovery.py, which provides a causal DAG for supporting further SHAP calculation.

3.Then do the SHAP calculations using algorithms in SHAP_explainer.py. 

4.The setting of machine learning methods for prediction is in the utils.py.

Other codes for small experiments of 3.4.1 and 3.4.2, or experiments on GeoSHAP:

Include codes for experiments on feature independence assumption(3.4.1), masked causes(3.4.2), and locational variable absorption effect(3.4.3), also for experiments on GeoShapley(5.1.1, 5.1.2)
