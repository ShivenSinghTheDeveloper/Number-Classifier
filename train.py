import numpy as np
import joblib
#train spli is for spliting data into training
#stratifiedK is used to make mini trains by spliting them but keeping the same digit balance
#corssval just returns scores
#Gridsearch trys different settings and it will choose the best one out of it
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from data import load_data
from evaluate import evaluate
from model import build_model
from visualize import show_samples, show_errors

def main():
    x,y,images= load_data()
    show_samples(images, y, n = 8)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,stratify=y, random_state=42)
    baseline = build_model(C=1.0, gamma="scale")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(baseline, x_train, y_train, cv=cv, scoring="accuracy")
    print("CV accuracy mean: ", cv_scores.mean())
    print("CV accuracy std: ", cv_scores.std())
    baseline.fit(x_train, y_train)
    baseline_preds = evaluate(baseline, x_test, y_test)
    show_errors(x_test, y_test, baseline_preds, n =12)
    print("\n===Grid Search===")
    param_grid = {
        #C is to see how strict is the model, and gamma is to see how detail is it
        #The idea of this section is combine these values to traing the model
        "svc__C":[0.5, 1, 2, 5, 10],
        "svc__gamma":["scale", 0.01, 0.03, 0.1]
    }
    
    grid = GridSearchCV(
        estimator = build_model(),
        param_grid=param_grid,
        cv = cv,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(x_train, y_train)
    print("Best params:", grid.best_params_)
    print("Best cv score:", grid.best_score_)
    
    
    #Setting up the best model and tuned predictions
    best_model = grid.best_estimator_
    joblib.dump(best_model, "digit_model.joblib")
    print("Model saved as digit model joblib")
    tuned_preds = evaluate(best_model, x_test, y_test)
    show_errors(x_test, y_test, tuned_preds, n=12)
    
    #Compare baseline vs tuned
    baseline_accuracy = (baseline_preds == y_test).mean()
    tuned_accuracy = (tuned_preds == y_test).mean()
    print("\n === Summary ====")
    print(f"Baseline test accuracy: {baseline_accuracy:.4f}")
    print(f"Tuned test accuracy: {tuned_accuracy:.4f}")
    
    #Check discord for the website example and docs.
if __name__ == "__main__":
    main()
    
    
    
#First we were succesfully able to run the code and shows number from 0 to 7 from our data set
#Now, its time to impement this same logic to read our phone pictures of handwriten pieces of paper
#Here are the step to deal with a picture (remember the concepts of gaussian blur that we did on the previous project)
""""
Steps:
    1) grayscale
    2) blur (reduce noise)
    3) adaptive threshold (handle uneven lighting)
    4) find the digit contour and crop
    5) pad to square + center
    6) resize to out_size
    7) scale to match sklearn digits range (0..16)
    8) flatten -> shape (64,) for 8x8
"""