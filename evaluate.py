import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

def evaluate(model, x_test, y_test):
    preds = model.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, preds))
    print("Classification Report: ", classification_report(y_test, preds))
    ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.title("Confusion Matrix")
    plt.show()
    
    return preds