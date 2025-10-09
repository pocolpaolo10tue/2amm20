from transformers import pipeline

# Load the model
detector = pipeline("text-classification", model="roberta-base-openai-detector")

# Run inference on a sample text
result = detector("IA MA A ROBOT BEEP BOOP")
print(result)
