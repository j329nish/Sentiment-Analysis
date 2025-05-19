# ライブラリの読み込み
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from sklearn.metrics import cohen_kappa_score
import torch
import numpy as np
import random
assert torch.cuda.is_available(), "You should use the GPU"

# 乱数の設定
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# トークナイザとモデルの設定
model_name = "tohoku-nlp/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# データセットの読み込み
dataset = load_dataset("llm-book/wrime-sentiment", remove_neutral=False, trust_remote_code=True)

# データの前処理
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)
train_dataset = dataset["train"].map(preprocess_function, batched=True)
eval_dataset = dataset["validation"].map(preprocess_function, batched=True)

# 訓練の設定
training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=1e-5,
    num_train_epochs=5,
    save_strategy="epoch",
    logging_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="qwk",
    fp16=torch.cuda.is_available()
)

# 評価関数の作成
def compute_qwk(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"qwk": cohen_kappa_score(labels, predictions)}

# Trainerのインスタンスを作成
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_qwk
)

# モデルの訓練
trainer.train()

# 最適なモデルの保存
trainer.save_model()