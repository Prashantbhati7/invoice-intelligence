from model_evaluation import train_random_forest,evaluate_classifier
from data_preprocessing import load_invoice_data,apply_label,split_data,scale_features
import joblib

FEATURES = ['invoice_quantity','invoice_dollars','Freight','total_item_quantity','total_item_dollars']

TARGET = 'flag_invoice_risk'

def main():

    df = load_invoice_data("/Users/prashant/Invoice Intelligence/data/inventory.db")
    df = apply_label(df)

    X_train, X_test, Y_train, Y_test = split_data(df,FEATURES,TARGET)
    X_train_scaled, X_test_scaled = scale_features(X_train,X_test)

    grid_search = train_random_forest(X_train_scaled,Y_train)
    evaluate_classifier(grid_search,X_test_scaled,Y_test,'Random Forest')

    joblib.dump(grid_search,'models/predict_flag_invoice.pkl')


if __name__ == "__main__":
    main()