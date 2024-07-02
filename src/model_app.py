from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb
import os

def round_two(val):
    return round(val,2)

class ModelApplication:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def _train_xgboost_classifier(self):
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.fit(self.X_train, self.y_train)

    def _get_prediction_metrics(self):
        y_pred = self.xgb_model.predict(self.X_test)

        mod_accuracy = accuracy_score(self.y_test.tolist(), y_pred.tolist())
        mod_precision = precision_score(self.y_test.tolist(), y_pred.tolist())
        mod_recall = recall_score(self.y_test.tolist(), y_pred.tolist())
        f1 = (2*mod_precision*mod_recall) / (mod_precision+ mod_recall)

        dict_metrics = {
            "Accuracy": round_two(mod_accuracy),
            "Precision": round_two(mod_precision),
            "Recall": round_two(mod_recall),
            "F1 Score" : round_two(f1)
        }

        return dict_metrics
    def evaluate_model(self):
        xgb_dict = self._get_prediction_metrics()

        return {  "XGBoost": xgb_dict}

    def get_model(self,force_retrain=False):
        path_model = 'data/xbgoost/model.json'
        if force_retrain:
            self._train_xgboost_classifier()
            self.xgb_model.save_model(path_model)
            print('Training is done')
            return
        if not os.path.exists(path_model) or os.stat(path_model).st_size == 0:
            print('XGBoost Model is training..')
            self._train_xgboost_classifier()
            self.xgb_model.save_model(path_model)
            print('Training is done')

        else:
            print('XGBoost Model is loading from disk..')
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(path_model)

    def model_predict(self,X_batch):
        return self.xgb_model.predict(X_batch)