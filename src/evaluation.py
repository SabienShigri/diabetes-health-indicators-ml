from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
import numpy as np

def evaluate_classification(model, X_test, y_test):
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))

def evaluate_regression(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE:", rmse)