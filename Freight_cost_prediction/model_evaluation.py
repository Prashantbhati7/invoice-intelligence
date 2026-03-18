from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def train_linear_regression(X_train, Y_train):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

def train_decision_tree(X_train, Y_train, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    model = DecisionTreeRegressor()
    model.fit(X_train, Y_train)
    return model

def train_random_forest(X_train, Y_train, n_estimators=100, max_depth=None, min_samples_split=4, min_samples_leaf=1):
    model = RandomForestRegressor()
    model.fit(X_train, Y_train)
    return model

def train_svm(X_train, Y_train, kernel='linear'):
    model = SVR()
    model.fit(X_train, Y_train)
    return model



def evaluate_model(model, X_test, Y_test,model_name) -> dict:
    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    print(f"model : {model_name}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print(f"Mean Absolute Error: {mae}")
    print()
    return {
        'model_name': model_name,
        'mse': mse,
        'r2': r2,
        'mae': mae
    }