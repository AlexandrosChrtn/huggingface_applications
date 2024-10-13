import torch
from transformers import pipeline

model_id = "AlexandrosChariton/SarcasLLM-1B"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float32,
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Should I move to Scandinavia?"},
]
outputs = pipe(
    messages,
    max_new_tokens=128
)
print(outputs[0]["generated_text"][-1])