from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def RF(estimators, max_depth, lr = None, task = "Regression"):
    if task == "Regression":
        forest = RandomForestRegressor(
            n_estimators=estimators,
            max_depth=max_depth
        )

    if task == "Binary":
        forest = RandomForestClassifier(
            n_estimators=estimators,
            max_depth=max_depth
        )

    return forest