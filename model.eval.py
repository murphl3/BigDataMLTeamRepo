from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
from data import preprocess_text  # assuming your preprocessing is in data.py
import pandas as pd

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("saved_model")
tokenizer = AutoTokenizer.from_pretrained("saved_model")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

def generate_recipe(input_ingredients: str):
    input_processed = preprocess_text(input_ingredients)
    prompt = f"Generate a structured recipe for: {input_processed}"
    encoded = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

# Example usage
example_input = "butter rice sugar "
print(generate_recipe(example_input))

import matplotlib.pyplot as plt

df = pd.read_csv("logs/recipe_model/version_0/metrics.csv")

# Extract losses (skip NaNs that may come from validation only being run at intervals)
train_losses = df[df["train_loss"].notna()]["train_loss"].tolist()

val_losses = [
    None, 1.510, 1.390, 1.330, 1.300, 1.270, 1.250, 1.240,
    1.230, 1.220, 1.220, 1.210, 1.210, 1.210, 1.220
]

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
