import pickle
import numpy as np
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
import gradio as gr

# Load the image embeddings index
with open('signatures.pkl', 'rb') as f:
    image_index = pickle.load(f)

# Initialize processor and model
processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
model = ViTModel.from_pretrained('facebook/dino-vitb16')

# Function to find the closest neighbor
def find_closest_neighbor(uploaded_image):
    # Preprocess the uploaded image
    image = Image.open(uploaded_image)
    inputs = processor(images=image, return_tensors="pt")
    
    # Get embeddings for the uploaded image
    with torch.no_grad():
        outputs = model(**inputs)
    # Apply mean pooling to aggregate token embeddings
    uploaded_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()  # Shape: (768,)
    
    # Vectorized cosine similarity computation
    image_paths = list(image_index.keys())
    embeddings = np.array(list(image_index.values()))  # Expected shape after pooling: (25, 768)
    
    # If image_index embeddings are not pooled, apply mean pooling
    if embeddings.ndim == 3:
        embeddings = embeddings.mean(axis=1)  # Shape: (25, 768)
    
    # Normalize embeddings
    uploaded_norm = np.linalg.norm(uploaded_embedding)
    embeddings_norm = np.linalg.norm(embeddings, axis=1)
    
    # Compute cosine similarities
    similarities = np.dot(embeddings, uploaded_embedding) / (embeddings_norm * uploaded_norm)
    
    # Find the index of the closest image
    closest_idx = np.argmax(similarities)
    closest_image_path = image_paths[closest_idx]
    closest_score = similarities[closest_idx]
    
    # Load the closest image
    closest_image = Image.open(closest_image_path)
    
    return closest_image, closest_score

# Create Gradio interface using the updated API
iface = gr.Interface(
    fn=find_closest_neighbor,
    inputs=gr.Image(type="filepath", label="Upload an Image"),
    outputs=[
        gr.Image(label="Closest Image"),
        gr.Number(label="Similarity Score")
    ],
    title="Image Similarity Search",
    description="Upload an image to find its closest match in our database!"
)

# Launch the app
iface.launch()
