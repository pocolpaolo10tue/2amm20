from transformers import pipeline

# Load the model
detector = pipeline("text-classification", model="roberta-base-openai-detector")

# Run inference on a sample text
result = detector("This is a sample text to check if it is AI-generated.")
print(result)
