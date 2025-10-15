import argparse, os, yaml
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
from seqeval.metrics import classification_report

def load_annotations(path):
    # Simple function: expect a .csv with columns text, entities (list of (start,end,label))
    import pandas as pd
    return pd.read_csv(path)

def create_token_class_dataset(df, tokenizer, labels_list):
    # Convert span markup to token-level BIO
    examples, features = [], []
    for _, row in df.iterrows():
        text = row['text']
        spans = [] # parse row['entities'] -> list of (start,end,label)
        # create labels per token -> BIO
        encoding = tokenizer(text, return_offsets_mapping=True, truncation=True)
        token_labels = ["O"] * len(encoding["offset_mapping"])
        for (s,e,lbl) in spans:
            # mark tokens whose offsets intersect span
            for i,(so,eo) in enumerate(encoding["offset_mapping"]):
                if so < e and eo > s:
                    token_labels[i] = "B-"+lbl if token_labels[i]=="O" else "I-"+lbl
        label_ids = [labels_list.index(l) for l in token_labels]
        examples.append({"input_ids": encoding["input_ids"], "attention_mask": encoding["attention_mask"], "labels": label_ids})
    return Dataset.from_list(examples)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--model_name", default="distilbert-base-cased")
    parser.add_argument("--out_dir", default="ner_model")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    labels = ["O"] + [f"B-{x}" for x in ["ANIMAL"]] + [f"I-ANIMAL"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    df = load_annotations(args.data_csv)
    ds = create_token_class_dataset(df, tokenizer, labels)

    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(labels))
    training_args = TrainingArguments(output_dir=args.out_dir, per_device_train_batch_size=8, num_train_epochs=args.epochs, evaluation_strategy="no")
    trainer = Trainer(model=model, args=training_args, train_dataset=ds)
    trainer.train()
    trainer.save_model(args.out_dir)

if __name__ == "__main__":
    main()
