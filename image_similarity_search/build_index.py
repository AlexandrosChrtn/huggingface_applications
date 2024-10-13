import os
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTModel
import pickle


# Initialize processor and model
processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
model = ViTModel.from_pretrained('facebook/dino-vitb16')

# Path to the image folder
image_folder = 'images/'

# Initialize the index dictionary
image_index = {}

# Loop through all image files in the folder
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_path = os.path.join(image_folder, image_file)
        
        # Open the image
        image = Image.open(image_path)
        
        # Preprocess the image and get embeddings
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get the last hidden state (embeddings) and store in index
        # Note: The embeddings are not L2 normalized!
        last_hidden_state = outputs.last_hidden_state
        image_index[image_path] = last_hidden_state.squeeze(0).numpy()  # Optionally convert to numpy for easier handling

# Fun fact: there are better ways to do the following! I don't want to 'annoy' you with details!
with open('signatures.pkl', 'wb') as f:
    pickle.dump(image_index, f)

# Print completion message
print(f"Index build successfully. Total embeddings: {len(image_index)}")
