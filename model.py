from sklearn.pipeline import Pipeline #chains together in the right order
from sklearn.preprocessing import StandardScaler #rescales the data to similar ranges
from sklearn.svm import SVC #how to tell digits apart

def build_model(c = 1.0, gamma = "scale"):
    return Pipeline([
        ("scaler", StandardScaler()), #This is used to only scale the data
        ("svc", SVC(kernel = "rbf", C=c, gamma=gamma)) #This is where the learn patterns are being applied
    ])