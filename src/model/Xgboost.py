from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def XGBoost(estimators, max_depth, lr, task = "Regression"):
    if task == "Regression":
        forest = XGBRegressor(
        n_estimators=estimators,
        max_depth=max_depth,
        learning_rate=lr
        )
        
    elif task == "Binary":
        forest = XGBClassifier(
        n_estimators=estimators,
        max_depth=max_depth,
        learning_rate=lr
        )
    return forest
