from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,accuracy_score,make_scorer,f1_score


def train_random_forest(X_train,Y_train):
    model = RandomForestClassifier(random_state=42,n_jobs=-1)
    param_grid = {
        'n_estimators': [100,200,300],
        'max_depth': [None,4,5,6],
        'min_samples_split': [2,3,5],
        'min_samples_leaf': [1,2,5],
        'criterion': ['gini','entropy']
    }
    scorer = make_scorer(f1_score)
    grid_search = GridSearchCV(estimator=model,param_grid=param_grid,cv=5,scoring=scorer,n_jobs=-1,verbose=0)
    grid_search.fit(X_train,Y_train)
    best_model = grid_search.best_estimator_
    return best_model

def evaluate_classifier(model,X_test,Y_test,model_name):
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test,Y_pred)
    report = classification_report(Y_test,Y_pred)
    print(f"{model_name} Performance")
    print(f"Accuracy: {accuracy:.2f}")
    print(report)
    print()
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'report': report
    }