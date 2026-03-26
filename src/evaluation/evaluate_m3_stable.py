import os
import sys
import time
import json
import torch
import re
import numpy as np
import wandb
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bert_score
from rouge_score import rouge_scorer

# Paths
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TEST_DATA_PATH = "/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_test.jsonl"


def load_test_data(path, max_samples=20):
    import random

    random.seed(42)
    data = []
    with open(path, "r") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except:
                continue
    random.shuffle(data)
    return data[:max_samples]


def extract_true_category(text):
    match = re.search(r"\[Category:\s*([^\]]+)\]", text)
    if match:
        cat = match.group(1).strip().lower()
        mapping = {
            "환경/위생": "environment",
            "도로/교통": "traffic",
            "시설물관리": "facilities",
            "민원서비스": "civil_service",
            "복지": "welfare",
            "기타": "other",
        }
        return mapping.get(cat, cat)
    return "other"


def parse_m3_category(response):
    ko_to_en = {
        "환경": "environment",
        "교통": "traffic",
        "도로": "traffic",
        "시설": "facilities",
        "민원": "civil_service",
        "복지": "welfare",
        "기타": "other",
    }
    clean = response.strip().lower()
    for ko, en in ko_to_en.items():
        if ko in clean:
            return en
    return "unknown"


def main():
    wandb.init(
        project="exaone-civil-complaint",
        name=f"m3-qwen-stable-eval-{datetime.now().strftime('%m%d-%H%M')}",
        config={"phase": "M3 Roadmap Final Stable", "model": MODEL_ID, "repetition_penalty": 1.1},
    )

    print(f"Loading {MODEL_ID} (4-bit)...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    model.eval()

    test_data = load_test_data(TEST_DATA_PATH, max_samples=20)

    print("Running M3 Roadmap Evaluation...")
    latencies, clean_gens, clean_refs, correct = [], [], [], 0

    for i, item in enumerate(test_data):
        prompt = f"당신은 민원 담당 AI입니다. 다음 민원에 친절하고 성실하게 답변하세요.\n\n민원: {item['input'][:500]}\n\n답변:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                repetition_penalty=1.1,
                do_sample=True,
                temperature=0.6,
            )
        latencies.append(time.perf_counter() - start)

        gen_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        clean_gens.append(gen_text.strip())
        clean_refs.append(item["output"].strip())

        # Classification
        class_prompt = f"다음 민원을 한 단어로 분류하세요: [환경, 교통, 시설, 민원, 복지, 기타]\n\n민원: {item['input'][:300]}\n결과:"
        c_in = tokenizer(class_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            c_out = model.generate(**c_in, max_new_tokens=10)
        c_text = tokenizer.decode(c_out[0][c_in.input_ids.shape[1] :], skip_special_tokens=True)

        if parse_m3_category(c_text) == extract_true_category(item["input"]):
            correct += 1

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l = np.mean(
        [scorer.score(r, g)["rougeL"].fmeasure * 100 for r, g in zip(clean_refs, clean_gens)]
    )
    P, R, F1 = bert_score.score(clean_gens, clean_refs, lang="ko")

    res = {
        "avg_latency": np.mean(latencies),
        "rouge_l": rouge_l,
        "bert_score_f1": F1.mean().item() * 100,
        "classification_accuracy": (correct / len(test_data)) * 100,
    }
    wandb.log(res)
    print(f"\nFinal M3 Results (Qwen Stable): {res}")
    wandb.finish()


if __name__ == "__main__":
    main()
