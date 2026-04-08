import gradio as gr
from test import predict

def classify_image(img):
    img.save("temp.jpg")
    return predict("temp.jpg")

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Inverse Prompt Generator",
    description="Upload an image and predict closest text description"
)

interface.launch()
