import joblib
import pandas as pd
import gradio as gr

# Load trained model and label encoder
model = joblib.load("iot_anomaly_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load dataset
data = pd.read_csv("webpagedata.csv")

# Function to make predictions
def predict_anomaly(input_row):
    input_df = pd.DataFrame([input_row])  # Convert to DataFrame
    numeric_prediction = model.predict(input_df)[0]  # Get numerical prediction
    category_prediction = label_encoder.inverse_transform([numeric_prediction])[0]  # Convert back to category
    return f"Predicted Category: {category_prediction}"

# Select row from CSV and predict
def select_row(index):
    row = data.iloc[index].tolist()
    return predict_anomaly(row)

# Gradio UI
iface = gr.Interface(fn=select_row, inputs=gr.Number(label="Row Index"), outputs="text")
iface.launch()
