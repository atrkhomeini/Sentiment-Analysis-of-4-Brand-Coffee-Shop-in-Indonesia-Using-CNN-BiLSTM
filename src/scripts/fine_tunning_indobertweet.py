# =============================
# Fine-tuning IndoBERTweet (3-Class Softmax)
# =============================

import pandas as pd
from datasets import Dataset

# Load dataset and encode label
df = pd.read_csv('../data/output/indobert_labeled_data.csv')
label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['Label_Bert'].str.lower().map(label2id)

dataset = Dataset.from_pandas(df[['Text', 'label']])
dataset = dataset.train_test_split(test_size=0.1)

# Load model and tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "indolem/indobertweet-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenize
def tokenize(example):
    return tokenizer(example['Text'], truncation=True, padding='max_length', max_length=128)

tokenized = dataset.map(tokenize, batched=True)

# Evaluation metrics
import evaluate
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=p.label_ids)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=p.label_ids, average='macro')["f1"]
    }

# Training arguments
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="../src/indobertweet/indobertweet-3class",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=2e-5,
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save model
model_name = 'indobertweet-finetuned-3class'
output_path = '../src/indobertweet/' + model_name
trainer.save_model(output_path)
tokenizer.save_pretrained(output_path)
