from transformers import pipeline
classifier = pipeline('zero-shot-classification', model='roberta-large-mnli', device="mps")

# See: https://huggingface.co/FacebookAI/roberta-large-mnli

sequence_to_classify = "Do you think that cereal belongs to the pizza?"
candidate_labels = ['travel', 'cooking', 'dancing', 'sport']
result = classifier(sequence_to_classify, candidate_labels)

# Pretty print the result
print(f"\nSequence: {result['sequence']}")
print("\nClassification Results:")
print("-----------------------")
for label, score in zip(result['labels'], result['scores']):
    print(f"{label:<10} : {score:.2%}")


# Candidate labels can be anything
sequence_to_classify = "Call Of Duty Vanguard PS4"
candidate_labels = ['playstation games', 'home and garden', 'pc games', 'mobile phones', 'beds', 'skateboards']
result = classifier(sequence_to_classify, candidate_labels)

# Pretty print the result
print(f"\nSequence: {result['sequence']}")
print("\nClassification Results:")
print("-----------------------")
for label, score in zip(result['labels'], result['scores']):
    print(f"{label:<10} : {score:.2%}")