from transformers import pipeline
import gradio as gr
# Gradio is the fastest way to demo your machine learning model with a friendly web interface so that anyone can use it, anywhere!

# See https://huggingface.co/briaai/RMBG-1.4
image_to_image = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

def generate_caption(image):
    result = image_to_image(image)
    return result

# Create a Gradio interface
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="🖼️ Remove BG 🪄",
    description="Upload an image, and watch as AI removes the background!"
)

# Launch the interface
iface.launch()