#!/usr/bin/env python3
# train.py

import os
import torch
import pandas as pd
import numpy as np
import evaluate  # 从独立包导入 evaluate

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

def main():
    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")
    if use_cuda:
        print(f"  GPU count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            print(f"    GPU {idx}: {torch.cuda.get_device_name(idx)}")
    else:
        print("  No GPU detected; using CPU.")


    # 1. 加载数据（BBC News Summary）
    data_url = (
        "https://raw.githubusercontent.com/"
        "sunnysai12345/News_Summary/master/news_summary_more.csv"
    )
    df = pd.read_csv(data_url)
    df = df.rename(columns={"text": "article", "headlines": "summary"})
    dataset = Dataset.from_pandas(df).train_test_split(test_size=0.1, seed=42)
    train_ds, val_ds = dataset["train"], dataset["test"]
    print(f"[Data] train: {len(train_ds)}, val: {len(val_ds)}")

    # 2. 准备模型和分词器
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 3. 预处理函数
    max_input_length, max_target_length = 512, 64
    def preprocess(examples):
        inputs  = examples["article"]
        targets = examples["summary"]
        model_inputs = tokenizer(inputs, max_length=max_input_length,
                                 truncation=True)
        labels = tokenizer(text_target=targets,
                           max_length=max_target_length,
                           truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_ds.map(preprocess, batched=True,
                             remove_columns=["article","summary"])
    val_tok   = val_ds.map(preprocess, batched=True,
                           remove_columns=["article","summary"])

    # 4. Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 5. ROUGE 评价
    rouge = evaluate.load("rouge")
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds,
                                               skip_special_tokens=True)
        labels = [[l if l != -100 else tokenizer.pad_token_id
                   for l in label] for label in labels]
        decoded_labels = tokenizer.batch_decode(labels,
                                                skip_special_tokens=True)
        results = rouge.compute(predictions=decoded_preds,
                                references=decoded_labels,
                                use_stemmer=True)
        return {k: round(v * 100, 4) for k, v in results.items()}

    # 6. 训练参数
    use_cuda = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir="bbc-sum-model",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_steps=500,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        weight_decay=0.01,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=use_cuda,
        report_to=[]
    )

    # 7. 初始化 Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 8. 开始训练并保存
    trainer.train()
    trainer.save_model("bbc-sum-model")
    tokenizer.save_pretrained("bbc-sum-model")
    print("👉 模型已保存到 bbc-sum-model/")

if __name__ == "__main__":
    main()
