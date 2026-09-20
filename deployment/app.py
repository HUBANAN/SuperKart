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

    sugar_map = {
        "Low Sugar": 0,
        "No Sugar": 1,
        "Regular": 2,
        "reg": 3
    }

    product_type_map = {
        "Baking Goods": 0,
        "Breads": 1,
        "Breakfast": 2,
        "Canned": 3,
        "Dairy": 4,
        "Frozen Foods": 5,
        "Fruits and Vegetables": 6,
        "Hard Drinks": 7,
        "Health and Hygiene": 8,
        "Household": 9,
        "Meat": 10,
        "Others": 11,
        "Seafood": 12,
        "Snack Foods": 13,
        "Soft Drinks": 14,
        "Starchy Foods": 15
    }

    store_size_map = {
        "High": 0,
        "Medium": 1,
        "Small": 2
    }

    city_map = {
        "Tier 1": 0,
        "Tier 2": 1,
        "Tier 3": 2
    }

    store_type_map = {
        "Departmental Store": 0,
        "Food Mart": 1,
        "Supermarket Type1": 2,
        "Supermarket Type2": 3
    }

    df = pd.DataFrame({
        "Product_Weight": [float(Product_Weight)],
        "Product_Sugar_Content": [sugar_map[Product_Sugar_Content]],
        "Product_Allocated_Area": [float(Product_Allocated_Area)],
        "Product_Type": [product_type_map[Product_Type]],
        "Product_MRP": [float(Product_MRP)],
        "Store_Establishment_Year": [float(Store_Establishment_Year)],
        "Store_Size": [store_size_map[Store_Size]],
        "Store_Location_City_Type": [city_map[Store_Location_City_Type]],
        "Store_Type": [store_type_map[Store_Type]]
    })
    
    prediction = model.predict(df)
    return f"Predicted Sales: {round(float(prediction[0]), 2)}"

demo = gr.Interface(
    fn=predict_sales,
   inputs=[
        gr.Number(label="Product Weight"),

        gr.Dropdown(
            ["Low Sugar", "No Sugar", "Regular", "reg"],
            label="Product Sugar Content"
        ),

        gr.Number(label="Product Allocated Area"),

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

    gr.Number(label="Product MRP"),

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
]
outputs=gr.Textbox(label="Prediction"),
title="SuperKart Sales Forecasting",
description="Predict Product Store Sales Total"
)

if __name__ == "__main__":
    demo.launch()
