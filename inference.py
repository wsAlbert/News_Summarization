import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def main():
    # 1. 检测 CUDA
    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")
    if use_cuda:
        print(f"  GPU count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            print(f"    GPU {idx}: {torch.cuda.get_device_name(idx)}")
    else:
        print("  No GPU detected; using CPU.")
    device = 0 if use_cuda else -1

    # 2. 加载已保存模型和分词器
    model_dir = "bbc-sum-model"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 3. 输出模型配置与参数量
    print("==== Model Configuration ====")
    print(model.config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Device set to use cuda:{device}")

    # 4. 从验证集中选取5条新闻作为示例
    print("== Loading validation set samples ==")
    full_dataset = load_dataset(
        "csv",
        data_files="https://raw.githubusercontent.com/sunnysai12345/News_Summary/master/news_summary_more.csv",
        split="train"
    )
    val_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)["test"]
    samples = val_dataset.shuffle(seed=42).select(range(5))

    # 5. 构建推理 pipeline
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    # 6. 展示摘要
    print("==== Summary Examples (from Validation Set) ====")
    for idx, item in enumerate(samples, 1):
        text = item["text"]  # 原始文章内容字段为 text
        print(f"Example {idx}:")
        print("Article:", text)
        # 动态设置摘要长度
        length = len(tokenizer(text).input_ids)
        max_len = max(10, min(60, length // 3))
        summary = summarizer(
            text,
            max_length=max_len,
            min_length=5,
            do_sample=False
        )[0]["summary_text"]
        print("Summary:", summary)

if __name__ == "__main__":
    main()

