from transformers import pipeline
import gradio as gr

# Gradio is the fastest way to demo your machine learning model with a friendly web interface so that anyone can use it, anywhere!
# Initialize the image-to-text pipeline
image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

def generate_caption(image):
    result = image_to_text(image)[0]
    return result['generated_text']

# Create a Gradio interface
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="🖼️ Magic Image Captioner 🪄",
    description="Upload an image, and watch as AI generates a caption for it!"
)

# Launch the interface
iface.launch()