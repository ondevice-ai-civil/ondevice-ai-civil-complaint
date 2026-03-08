
import os
import sys
import time
import json
import torch
import numpy as np
import wandb
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import bert_score

# Paths
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TEST_DATA_PATH = "/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_test.jsonl"

def main():
    wandb.init(project="exaone-civil-complaint", name=f"m3-qwen-final-{datetime.now().strftime('%Y%m%d-%H%M')}")

    print(f"Loading {BASE_MODEL_ID} (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    
    # Data
    test_data = []
    with open(TEST_DATA_PATH, 'r') as f:
        for line in f: test_data.append(json.loads(line))
    test_data = test_data[:10] 

    print("Running Eval...")
    latencies = []
    clean_gens = []
    clean_refs = []
    
    for item in test_data:
        prompt = f"당신은 민원 담당 AI입니다. 다음 민원에 친절하게 답변하세요.\n\n민원: {item['input'][:500]}\n\n답변:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, repetition_penalty=1.1)
        latencies.append(time.perf_counter()-start)
        gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Gen: {gen[:50]}...")
        clean_gens.append(gen.strip())
        clean_refs.append(item['output'].strip())

    P, R, F1 = bert_score.score(clean_gens, clean_refs, lang="ko")
    metrics = {"avg_latency": np.mean(latencies), "bert_score_f1": F1.mean().item() * 100}
    wandb.log(metrics)
    print(f"Final Metrics: {metrics}")
    wandb.finish()

if __name__ == "__main__":
    main()
