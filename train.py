from collections import defaultdict

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import AdamW, BertModel, get_linear_schedule_with_warmup

import config
from engine import eval_fn, train_fn
from model import SentimentClassifier
from utils import create_data_loader, rating_to_sentiment

class_labels = ["negative", "neutral", "positive"]


def run():
    df = pd.read_csv("inputs/reviews.csv")
    df["sentiment"] = df.score.apply(rating_to_sentiment)
    df_train, df_rem = train_test_split(
        df, test_size=0.1, random_state=config.RANDOM_SEED
    )
    df_val, df_test = train_test_split(
        df_rem, test_size=0.5, random_state=config.RANDOM_SEED
    )
    train_data_loader = create_data_loader(
        df_train, config.TOKENIZER, config.MAX_LEN, config.BATCH_SIZE
    )
    val_data_loader = create_data_loader(
        df_val, config.TOKENIZER, config.MAX_LEN, config.BATCH_SIZE
    )
    test_data_loader = create_data_loader(
        df_test, config.TOKENIZER, config.MAX_LEN, config.BATCH_SIZE
    )

    # data = next(iter(val_data_loader))
    # input_ids = data["input_ids"].to(config.DEVICE)
    # attention_mask = data["attention_mask"].to(config.DEVICE)
    # bert_model = BertModel.from_pretrained(config.BERT_NAME)

    model = SentimentClassifier(num_classes=len(class_labels))
    if config.LOAD_MODEL == True:
        model.load_state_dict(torch.load("best_model_state.bin"))
    model = model.to(config.DEVICE)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(config.DEVICE)

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        print("-" * 10)

        train_acc, train_loss = train_fn(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            config.DEVICE,
            scheduler,
            len(df_train),
        )

        print(f"Train loss {train_loss} accuracy {train_acc}")

        val_acc, val_loss = eval_fn(
            model, val_data_loader, loss_fn, config.DEVICE, len(df_val)
        )

        print(f"Val   loss {val_loss} accuracy {val_acc}")
        print()

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), "best_model_state.bin")
            best_accuracy = val_acc


if __name__ == "__main__":
    run()
