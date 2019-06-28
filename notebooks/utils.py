from sklearn.metrics import accuracy_score, mean_squared_error

def evaluate_classifier(clf, X_test, y_test):
    y_test_pred, y_test_true = clf.predict(X_test), y_test
    return accuracy_score(y_test_true, y_test_pred)

def evaluate_regressor(model, X_test, y_test):
    y_test_pred, y_test_true = model.predict(X_test), y_test
    return mean_squared_error(y_test_true, y_test_pred)
