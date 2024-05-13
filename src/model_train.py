from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score

def round_two(val):
    return round(val,2)

def encode_label(labels:list):
    out = []
    for i in range(len(labels)):
        if labels[i] == 'approved':
            out.append(1)
        else:
            out.append(0)
    return out

class ModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def _train_sgd_classifier(self):
        self.sgd_model = SGDClassifier()
        self.sgd_model.fit(self.X_train, self.y_train)
    def _train_decision_tree(self):
        self.dt_model = DecisionTreeClassifier()
        self.dt_model.fit(self.X_train, self.y_train)

    def _train_neural_network(self):
        self.nn_model = MLPClassifier()
        self.nn_model.fit(self.X_train, self.y_train)

    def _train_lightgbm(self):
        self.lgbm_model = LGBMClassifier()
        self.lgbm_model.fit(self.X_train, self.y_train)

    def _train_xgboost_classifier(self):
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.fit(self.X_train, self.y_train)

    def _get_prediction_metrics(self,model):
        y_pred = model.predict(self.X_test)
        mod_accuracy = accuracy_score(self.y_test, y_pred)
        mod_precision = precision_score(self.y_test, y_pred)
        mod_recall = recall_score(self.y_test, y_pred)

        dict_metrics = {
            "Accuracy": round_two(mod_accuracy),
            "Precision": round_two(mod_precision),
            "Recall": round_two(mod_recall)}

        return dict_metrics

    def train_models(self):
        self._train_lightgbm()
        self._train_decision_tree()
        self._train_neural_network()
        self._train_sgd_classifier()
        self._train_xgboost_classifier()

        print('\n\nLightGBM, Decision Tree, SGD and Neural Network are trained on dataset.')

    def evaluate_models(self):
        dt_dict   = self._get_prediction_metrics(self.dt_model)
        lgbm_dict = self._get_prediction_metrics(self.lgbm_model)
        nn_dict   = self._get_prediction_metrics(self.nn_model)
        sgd_dict  = self._get_prediction_metrics(self.sgd_model)
        xgb_dict  = self._get_prediction_metrics(self.xgb_model)

        return {
            "Decision Tree" : dt_dict,
            "LightGBM" : lgbm_dict,
            "SGD Classifier" : sgd_dict,
            "XGBoost" : xgb_dict,
            "Neural Network" : nn_dict
        }

    def get_decision_tree_feature_importance(self):
        return self.dt_model.feature_importances_

    def _get_lightgbm_feature_importance_sub(self):
        return self.lgbm_model.feature_importances_

    def _normalize_feature_importance(self, importance_values):
        total_importance = np.sum(importance_values)
        normalized_importance = importance_values / total_importance
        return normalized_importance

    def get_lightgbm_feature_importance(self):
        lgbm_feature_importance = self._get_lightgbm_feature_importance_sub()
        normalized_importance = self._normalize_feature_importance(lgbm_feature_importance)
        return normalized_importance

    def get_neural_network_feature_importance(self):

        coefficients = self.nn_model.coefs_[0]
        absolute_weights = np.abs(coefficients)

        total_absolute_weight = np.sum(absolute_weights)
        normalized_importance = absolute_weights / total_absolute_weight * 100
        return normalized_importance

    def get_sgd_classifier_feature_importance(self):
        coefficients = self.sgd_model.coef_
        absolute_weights = np.abs(coefficients)

        total_absolute_weight = np.sum(absolute_weights)
        normalized_importance = absolute_weights / total_absolute_weight *100

        return normalized_importance

    def get_xgboost_feature_importance(self):
        return self.xgb_model.feature_importances_

    def get_feature_importance(self):
        dt_feats = self.get_decision_tree_feature_importance()
        lgbm_feats = self.get_lightgbm_feature_importance()
        dnn_feats = self.get_neural_network_feature_importance()
        sgd_feats = self.get_sgd_classifier_feature_importance()
        xgb_feats = self.get_xgboost_feature_importance()

        return dt_feats, lgbm_feats, dnn_feats,sgd_feats, xgb_feats