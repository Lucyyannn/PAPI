import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def process_dolly_for_papi(output_file="dolly.csv"):

    # 加载数据集
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    target_categories = ["creative_writing", "open_qa"]
    
    # 初始化 Tokenizer 
    try:
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    except Exception as e:
        print(f"加载失败，尝试备用路径: {e}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 统计数据信息
    processed_data = []

    for entry in tqdm(dataset):
        category = entry["category"]
        if category in target_categories:
            input_text = entry["instruction"]
            if entry["context"]:
                input_text += "\n" + entry["context"]
            
            output_text = entry["response"]

            # 分词
            lin_tokens = tokenizer.encode(input_text)
            lout_tokens = tokenizer.encode(output_text)

            processed_data.append({
                "category": category,
                "lin": len(lin_tokens),
                "lout": len(lout_tokens)
            })

    df = pd.DataFrame(processed_data)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    process_dolly_for_papi()