from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont

# See https://huggingface.co/facebook/detr-resnet-101
path_to_img = "image.jpg"
image = Image.open(path_to_img)

# You can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Get image dimensions to adjust font size and bounding box thickness
image_width, image_height = image.size
font_size = int(min(image_width, image_height) * 0.03)  # Font size is 3% of the smaller dimension
box_thickness = int(min(image_width, image_height) * 0.005)  # Box thickness is 0.5% of the smaller dimension

# Convert outputs (bounding boxes and class logits) to COCO API
# Let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# Create a copy of the image to draw on
image_with_boxes = image.copy()
draw = ImageDraw.Draw(image_with_boxes)

# Load a better font or fallback to default if a font is not available
try:
    font = ImageFont.truetype("arial.ttf", font_size)
except IOError:
    font = ImageFont.load_default()

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    
    # Draw bounding box with a dynamic thickness
    draw.rectangle(box, outline="red", width=box_thickness)
    
    # Prepare label text
    label_text = f"{model.config.id2label[label.item()]}: {score.item():.2f}"
    
    # Calculate text size
    text_bbox = draw.textbbox((0, 0), label_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Calculate position for text and background
    text_x = box[0]
    text_y = box[3]  # Place text below the box
    
    # Draw filled background for text
    draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill="white")
    
    # Draw text on the filled background
    draw.text((text_x, text_y), label_text, fill="black", font=font)

    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{score.item():.2f} at location {box}"
    )

# Save the image with bounding boxes
image_with_boxes.save("output_image_with_boxes.jpg")
print("Image with bounding boxes saved as 'output_image_with_boxes.jpg'")
