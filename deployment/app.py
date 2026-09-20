import gradio as gr
 
def predict(text):
return f"You entered: {text}"
 
demo = gr.Interface(
fn=predict,
inputs="text",
outputs="text",
title="SuperKart"
)
 
demo.launch()
