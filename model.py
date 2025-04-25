import os
import ast
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,T5ForConditionalGeneration
import nltk
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.optim import AdamW
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import Dataset
import kagglehub
import torch
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Download the latest version of the RecipeNLG dataset
path = kagglehub.dataset_download("paultimothymooney/recipenlg")

# Show the path where the dataset was downloaded
print("Path to dataset files:", path)

# Optional: list the contents of the dataset directory
print("Files in the dataset:")
print(os.listdir(path))

# Full path to the CSV file
csv_path = os.path.join(path, "RecipeNLG_dataset.csv")

# Load the CSV into a DataFrame
df = pd.read_csv(csv_path)


df_sample = df.sample(n=10000, random_state=1010).reset_index(drop=True)

# Show the shape and a preview
print("Dataset shape:", df_sample.shape)
# printing only title, ingredients, directions, and NER
(df_sample[['title', 'ingredients', 'directions', 'NER']].head())

# Lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")).union({
    "recipe", "want", "need", "make", "cook", "prepare", "food", "with",
    "and", "create", "dish", "meal"
})

def preprocess_ingredients(ingredients):
    try:
        items = ast.literal_eval(ingredients) if ingredients.startswith("[") else [ingredients]
        return "\n".join(f"- {item.strip()}" for item in items if item.strip())
    except:
        return str(ingredients)

def preprocess_directions(directions):
    try:
        items = ast.literal_eval(directions) if directions.startswith("[") else [directions]
        return "\n".join(f"{i + 1}. {step.strip()}" for i, step in enumerate(items) if step.strip())
    except:
        return str(directions)

def preprocess_text(text):
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    return " ".join([lemmatizer.lemmatize(w) for w in words if w not in stop_words]) or "unknown"

def format_recipe(row):
    return {
        "input_text": f"Generate a structured recipe for: {row['processed_ingredients']}",
        "target_text": f"Title: {row['title']}\nIngredients:\n{row['ingredients']}\nDirections:\n{row['directions']}\n"
    }

def get_dataloaders(df, batch_size=4):
    # Preprocess
    df['ingredients'] = df['ingredients'].apply(preprocess_ingredients)
    df['directions'] = df['directions'].apply(preprocess_directions)
    df['processed_ingredients'] = df['ingredients'].apply(preprocess_text)

    # Format data
    df = pd.DataFrame([format_recipe(row) for _, row in df.iterrows()])

    # Split into train, validation, and test sets
    df_train_val, df_test = train_test_split(df, test_size=0.1, random_state=42)
    df_train, df_val = train_test_split(df_train_val, test_size=0.15, random_state=42)

    # Convert to HuggingFace Datasets
    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_val)
    test_dataset = Dataset.from_pandas(df_test)

    # Tokenize datasets
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            padding="max_length",
            truncation=True,
            max_length=128  # input prompt only needs ingredients
        )
        labels = tokenizer(
            text_target=examples["target_text"],
            padding="max_length",
            truncation=True,
            max_length=256
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Apply tokenization
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["input_text", "target_text"])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["input_text", "target_text"])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["input_text", "target_text"])

    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, tokenizer

# Define your PyTorch Lightning model
class SimpleRecipeT5(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")  # Or t5-base
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr)

csv_logger = CSVLogger("logs", name="recipe_model")

# Initialize Trainer with GPU acceleration (mps, if you're on M1/M2 mac)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./checkpoints",
    filename="simple-t5-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
    mode="min"
)

trainer = pl.Trainer(
    accelerator="cuda" if torch.cuda.is_available() else "cpu",
    max_epochs=15,
    log_every_n_steps=10,
    callbacks=[checkpoint_callback],
    logger=csv_logger,
    precision=16 if torch.cuda.is_available() else 32,
    gradient_clip_val=1.0,
    limit_train_batches=0.5,  # Use more data for training
    limit_val_batches=0.2,  # Keep the validation fraction
    accumulate_grad_batches=2
)

# Load Data
df_small = df_sample.sample(5000)  # Reduce dataset to ~5000 rows for testing
batch_size = 8
train_loader, val_loader, test_loader, tokenizer = get_dataloaders(df_small, batch_size=batch_size)



# Train the model
model = SimpleRecipeT5()
trainer.fit(model, train_loader, val_loader)

# Save the model and tokenizer
model.model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")

# Recipe Generation
def generate_recipe(input_ingredients: str):
    input_processed = preprocess_text(input_ingredients)

    # Format prompt for recipe generation
    prompt = f"Generate a structured recipe for: {input_processed}"

    # Tokenize the prompt
    encoded = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)

    # Generate recipe from the model
    outputs = model.model.generate(
        **encoded,
        max_length=256,
        num_beams=4,
        early_stopping=True
    )

    # Decode the output
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean up the output to ensure structured recipe format
    if "Title:" in decoded and "Ingredients:" in decoded and "Directions:" in decoded:
        return decoded
    else:
        return f"Generated Recipe:\n{decoded}"

# Example usage
example_input = "tuna cheese peppers "
print(generate_recipe(example_input))

# Plotting training and validation loss
log_path = csv_logger.log_dir
metrics_df = pd.read_csv(os.path.join(log_path, "metrics.csv"))

# Drop rows with NaNs to avoid plotting issues
metrics_df = metrics_df.dropna(subset=["train_loss", "val_loss"], how="all")

plt.figure(figsize=(10, 5))
plt.plot(metrics_df["step"], metrics_df["train_loss"], label="Train Loss")
plt.plot(metrics_df["step"], metrics_df["val_loss"], label="Validation Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()