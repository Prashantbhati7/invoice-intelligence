import joblib
from pathlib import Path
import pickle
from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from model_evaluation import (train_linear_regression, train_decision_tree, train_random_forest, train_svm, evaluate_model)


def main():
    db_path='/Users/prashant/Invoice Intelligence/data/inventory.db'
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    df = load_vendor_invoice_data(db_path)
    X,Y = prepare_features(df)
    X_train, X_test, Y_train, Y_test = split_data(X,Y)
    lr_model = train_linear_regression(X_train, Y_train)
    dt_model = train_decision_tree(X_train, Y_train)
    rf_model = train_random_forest(X_train, Y_train)
    svm_model = train_svm(X_train, Y_train)
    results =[]
    results.append(evaluate_model(lr_model, X_test, Y_test,'Linear Regression'))
    results.append(evaluate_model(dt_model, X_test, Y_test,'Decision Tree'))
    results.append(evaluate_model(rf_model, X_test, Y_test,'Random Forest'))
    results.append(evaluate_model(svm_model, X_test, Y_test,'SVM'))
    print(results)
    print()
    best_model_info = min(results, key=lambda x: x['mae'])
    best_model_name = best_model_info['model_name']
    all_models = {'Linear Regression': lr_model, 'Decision Tree': dt_model, 'Random Forest': rf_model, 'SVM': svm_model}
    best_model = all_models[best_model_name]
    print()
    print(f"Best Model: {best_model_name}")

    model_path = model_dir / f'{best_model_name}.pkl'
    pickle.dump(best_model, open(model_path, 'wb'))

if __name__ == "__main__":
    main()