import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH =  'models/predict_flag_invoice.pkl'

def load_model(model_path:str = MODEL_PATH):
    """" Load Trained freight cost prediction model """
    with open(model_path,'rb') as f:
        model = joblib.load(f)

    return model


def predict_flag_invoice(input_data):
    """ Predict freight cost for new vendor invoices
        Parameters
        -----------
        input data : dict

        Returns
        ----------
        pd.Dataframe with Predicted freight cost
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Flag'] = model.predict(input_df).round()
    return input_df 


if __name__ == "__main__":
    sample_data = {
        "invoice_quantity":[10],
        "invoice_dollars":[1000],
        "Freight":[100],
        "total_item_quantity":[10],
        "total_item_dollars":[1000],
    }
    prediction = predict_flag_invoice(sample_data)
    print(prediction)