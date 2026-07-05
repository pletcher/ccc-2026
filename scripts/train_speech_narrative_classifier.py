from sklearn.utils import compute_class_weight
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from torch.optim import Adam
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


ROOT_DIR = Path(__file__).parent.parent
IN_CSV = ROOT_DIR / "csv" / "homer_speech_and_narrative_by_sentence.csv"
OUTPUT_DIR = ROOT_DIR / "speech-narrative_classification_model_output"

labels = {"narrative": 0, "speech": 1}
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

df = pd.read_csv(IN_CSV)
df["label"] = df.register.replace(labels)

model_path = "pranaydeeps/Ancient-Greek-BERT"

tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"

num_epochs = 2
lr = 0.00001
batch_size = 32

train_df, test_df = sklearn_train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

print(f"Train size = {len(train_df)}; test size = {len(test_df)}")

train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
test_dataset = Dataset.from_pandas(test_df[["text", "label"]])


def tokenize_fn(rows):
    return tokenizer(
        rows["text"], padding="max_length", truncation=True, max_length=500
    )  # ty:ignore[call-non-callable]


train_dataset = train_dataset.map(tokenize_fn, batched=True)
test_dataset = test_dataset.map(tokenize_fn, batched=True)


class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_dataset["label"]),
    y=np.array(train_dataset["label"]),
)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

config = AutoConfig.from_pretrained(
    model_path,
    attention_probs_dropout_prob=0.1,
    hidden_dropout_prob=0.1,
    id2label=id2label,
    label2id=label2id,
    num_labels=2,
    random_seed=42,
)

model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)

optimizer = Adam(model.parameters(), lr=lr)

model.to(device)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=num_epochs,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=50,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

trainer.save_model(str(OUTPUT_DIR / "tire-kicking"))
