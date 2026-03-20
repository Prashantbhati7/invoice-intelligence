import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH =  'models/predict_flag_invoice.pkl'
SCALER = 'models/scaler.pkl'
def load_model(model_path:str = MODEL_PATH):
    """" Load Trained freight cost prediction model """
    with open(model_path,'rb') as f:
        model = joblib.load(f)
    return model

def load_scaler(scaler_path:str = SCALER):
    with open(scaler_path,'rb') as f:
        scaler = joblib.load(f)
    return scaler

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
    print(f"input df before scaling {input_df}")
    scaler = load_scaler()
    input_df = scaler.transform(input_df)
    # print(f"input df after scaling \n {input_df}")
    result = model.predict(input_df).round()
    return result


if __name__ == "__main__":
    sample_data = {
        "invoice_quantity":[6],
        "invoice_dollars":[214.26],
        "Freight":[3.47],
        "total_item_quantity":[6],
        "total_item_dollars":[214.26],
    }
    prediction = predict_flag_invoice(sample_data)
    print()
    print(prediction)