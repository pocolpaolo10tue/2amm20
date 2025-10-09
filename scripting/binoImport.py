from binoculars.detector import Binoculars

# Create the model (this will download Falcon-7B and Falcon-7B-Instruct)
bino = Binoculars(
    observer_name_or_path="tiiuae/falcon-7b",
    performer_name_or_path="tiiuae/falcon-7b-instruct",
    mode="accuracy",  # or "low-fpr"
)

# Run prediction
text = "The quick brown fox jumps over the lazy dog."
result = bino.predict(text)

print("Prediction:", result)
