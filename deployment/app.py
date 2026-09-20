import gradio as gr
import pandas as pd
import joblib

from huggingface_hub import hf_hub_download

# Load model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="ANNELEARN/SuperKart-model",
    filename="best_superkart_model.pkl"
)

model = joblib.load(model_path)

def predict_sales(
    Product_Weight,
    Product_Sugar_Content,
    Product_Allocated_Area,
    Product_Type,
    Product_MRP,
    Store_Establishment_Year,
    Store_Size,
    Store_Location_City_Type,
    Store_Type
):

    df = pd.DataFrame({
        "Product_Weight": [float(Product_Weight)],
        "Product_Sugar_Content": [float(Product_Sugar_Content)],
        "Product_Allocated_Area": [float(Product_Allocated_Area)],
        "Product_Type": [float(Product_Type)],
        "Product_MRP": [float(Product_MRP)],
        "Store_Establishment_Year": [float(Store_Establishment_Year)],
        "Store_Size": [float(Store_Size)],
        "Store_Location_City_Type": [float(Store_Location_City_Type)],
        "Store_Type": [float(Store_Type)]
    })

    prediction = model.predict(df)

    return f"Predicted Sales: {round(float(prediction[0]), 2)}"

demo = gr.Interface(
    fn=predict_sales,
    inputs=[
        gr.Number(label="Product Weight"),
        gr.Number(label="Product Allocated Area"),
        gr.Number(label="Product MRP"),
        gr.Dropdown(
            ["Low Sugar", "No Sugar", "Regular", "reg"],
            label="Product Sugar Content"
        ),

        gr.Dropdown(
            [
                "Baking Goods",
                "Breads",
                "Breakfast",
                "Canned",
                "Dairy",
                "Frozen Foods",
                "Fruits and Vegetables",
                "Hard Drinks",
                "Health and Hygiene",
                "Household",
                "Meat",
                "Others",
                "Seafood",
                "Snack Foods",
                "Soft Drinks",
                "Starchy Foods"
            ],
            label="Product Type"
        ),   
        gr.Number(label="Store Establishment Year"),
        gr.Dropdown(
                ["High", "Medium", "Small"],
                label="Store Size"
        ),
        gr.Dropdown(
                ["Tier 1", "Tier 2", "Tier 3"],
                label="Store Location City Type"
        ),
        gr.Dropdown(
            [
                "Departmental Store",
                "Food Mart",
                "Supermarket Type1",
                "Supermarket Type2"
            ],
                label="Store Type"
            )
    ],
outputs=gr.Textbox(label="Prediction"),
title="SuperKart Sales Forecasting",
description="Predict Product Store Sales Total"
)

if __name__ == "__main__":
    demo.launch()
