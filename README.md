# huggingface applications

A collection of really cool Hugging Face applications that you can easily run locally.

Presented at the PyThess meetup, this repository showcases various machine learning tools built using Hugging Face models. Below is an overview of its contents, along with instructions on how to use each tool and the models they utilize. Feel free to get back to me with any questions, suggestions and anything else.

## Table of Contents

- [LLM Text Generation](#llm-text-generation)
- [Image Captioning](#image-captioning)
- [Image Object Detection](#image-object-detection)
- [Sound to Text](#sound-to-text)
- [Zero Shot Classification](#zero-shot-classification)
- [Image Background Removal](#image-background-removal)
- [Image Similarity Search](#image-similarity-search)

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python. Ask chatgpt for help.
- [pip](https://pip.pypa.io/en/stable/installation/)
- Git (optional, for cloning the repository)
- Cuda (Only for non-Macbook users, optional for most examples, for running stuff on GPUs)
### Installation
1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/huggingface_applications.git
   cd huggingface_applications
   ```

2. **Create a Virtual Environment**

   It's recommended to create a virtual environment to manage dependencies.

   ```bash
   python3 -m venv huggingface_apps
   source huggingface_apps/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Tools Overview

   Change directory from terminal with:
   
   ```bash
   cd name_of_folder
   ```
   where name_of_folder is the name of the folder containing the tool you want to run.

   ALWAYS remember, if the model is not downloaded locally, give it time to download! This is handled automatically by our beloved Hugging Face!
### LLM Text Generation

Leverages large language models to generate coherent and contextually relevant text based on user input.

- **Model Used:** [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and our slick and sarcastic fine-tuned model [`AlexandrosChariton/SarcasMLL-1B`](https://huggingface.co/AlexandrosChariton/SarcasMLL-1B)

#### Usage
Do not bother with this unless you have decent hardware and Nvidia cuda or Apple's MPS
1. **Load and Use Pretrained Model**
    If you want to use any pretrained model, run the following command with the appropriate model id:
   ```bash
   cd text_generation
   python3 generate_text.py
   ```

2. **Fine-Tune Your Own Model**
   Do not bother unless you're rich in computational resources. Open `train_our_own_fine_tune.ipynb` in Jupyter Notebook and follow the steps to fine-tune the model on your dataset.

3. **Load our Fine-Tuned Model from Hub**
    During the meeting we showed a sarcastic assistant who does not really help with anything. Was built on top of Llama-3.2-1B-Instruct. Add whatever text you want to ```load_our_model_from_hub.py``` and generate!
   ```bash
   python3 load_our_model_from_hub.py
   ```
   Example input:
   ```"Should I move to Scandinavia?"```

   Response:
   ```{'role': 'assistant', 'content': "Oh yes, because nothing says 'good life' like freezing your butt off. And the cost of living? A whole other story. You might even need a warm coat. Worth a shot? Probably not. Scandinavia is all about embracing the cold. You'll love it. You'll hate it. Either way, you'll be fine. Or not. Who knows. It's all part of the adventure. Right?"}```

### Image Captioning

Generates descriptive captions for input images using state-of-the-art image-to-text models.

- **Model Used:** [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base)

#### Usage
   ```bash
   cd image_to_text
   python3 demo_gradio_image_to_text.py
   ```
   Visit the URL provided by the script to use the application!
   A Gradio interface will launch, allowing you to upload an image and receive a generated caption.

### Image Object Detection

Detects and labels objects within images, drawing bounding boxes around identified items.

- **Model Used:** [`facebook/detr-resnet-101`](https://huggingface.co/facebook/detr-resnet-101)

The script processes a predefined image (`image.jpg`), detects objects, and saves the output image with bounding boxes as `output_image_with_boxes.jpg`. The objects found and their bounding boxes coordinates are printed to the terminal. If you want to see how it performs, place your image named (`image.jpg`) inside the directory and run the script with
   ```bash
   cd object_detection
   python3 detect_objects.py
   ```

It also works with .webp and .png, but you have to change the variable ```path_to_img``` inside ```detect_objects.py``` to the path of the image you want to use.


### Sound to Text

Transcribes spoken language from audio recordings into written text. You can pretty much say something and use it as a python string!

- **Model Used:** [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3)

#### Usage
You can either opt for the Flask API which opens a browser window and records sound with a button or the ```transcribe_audio.py``` which records audio from your microphone and outputs the transcription to the console.
1. **Run the Flask Application**

   ```bash
   python app.py
   ```

   Navigate to `http://localhost:5000` in your browser to access the web interface for recording audio. 

2. **Direct Audio Transcription**

   ```bash
   python sound_to_text/transcribe_audio.py
   ```

   This script records audio for a specified duration and outputs the transcription in the console.

### Zero Shot Classification

Classifies text into predefined categories without requiring any task-specific training. Set the labels you want to classify your text into in the ```candidate_labels``` variable in the ```zero_shot_classification_demo.py``` file and the text you want to classify in the ```sequence_to_classify``` variable! Enjoy!

- **Model Used:** [`roberta-large-mnli`](https://huggingface.co/FacebookAI/roberta-large-mnli)

#### Usage
```bash
cd zero_shot_classification
python3 zero_shot_classification_demo.py
```
### Image Background Removal
Removes the background from images, leaving only the main subject.

- **Model Used:** [`briaai/RMBG-1.4`](https://huggingface.co/briaai/RMBG-1.4)

#### Usage
```bash
cd remove_background
python3 demo_gradio_remove_bg.py
```
Then follow the url provided by the script to use the application!

A Gradio interface will launch, allowing you to upload an image and receive the background-removed version. Enjoy editing your images to remove unwanted people!

### Image Similarity Search

Finds and displays images similar to a given input image from a predefined database.

- **Model Used:** [`facebook/dino-vitb16`](https://huggingface.co/facebook/dino-vitb16)

#### Setup
1. Make sure you have the images you want to search through in the ```images/``` folder.

2. **Build the Image Embeddings Index**

   ```bash
   cd image_similarity_search
   python build_index.py
   ```
   This script processes images in the `images/` directory and creates an embeddings index (`signatures.pkl`).

3. **Run the Gradio Application**

   ```bash
   python gradio_app.py
   ```

   A Gradio interface will launch, allowing you to upload an image and find its closest match in the database! Feel free to change the model if you want, but remember, the model choice needs to be consistent across index building and generating signatures for query images.

Fun quiz: Assuming we wanted to identify people based on their attire, could we use this image search to do that? Is there anything else in this codebase that can help with that? Stay tuned!