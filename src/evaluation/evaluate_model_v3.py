
import os
import sys
import time
import json
import torch
import re
import numpy as np
import wandb
from datetime import datetime
from collections import defaultdict
from vllm import LLM, SamplingParams
import bert_score
from rouge_score import rouge_scorer

# Monkey-patch for EXAONE compatibility with latest transformers if needed
import transformers.utils.generic
if not hasattr(transformers.utils.generic, 'check_model_inputs'):
    transformers.utils.generic.check_model_inputs = lambda *args, **kwargs: None

# Paths
MODEL_DIR = "/content/ondevice-ai-civil-complaint/models/merged_model"
TEST_DATA_PATH = "/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_test.jsonl"
RESULTS_DIR = "/content/ondevice-ai-civil-complaint/docs/outputs/M3_Optimization"

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_test_data(path, max_samples=100):
    import random
    random.seed(42)
    data = []
    with open(path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError:
                continue
    random.shuffle(data)
    return data[:max_samples]

def extract_true_category(text):
    """Extract category from input text [Category: xxx] pattern."""
    match = re.search(r'\[Category:\s*([^\]]+)\]', text)
    if match:
        cat = match.group(1).strip().lower()
        mapping = {
            "환경/위생": "environment", "도로/교통": "traffic", "시설물관리": "facilities",
            "민원서비스": "civil_service", "복지": "welfare", "기타": "other",
            "environment": "environment", "traffic": "traffic", "facilities": "facilities", "other": "other"
        }
        return mapping.get(cat, cat)
    return "unknown"

def parse_predicted_category(response):
    ko_to_en = {
        "환경": "environment", "위생": "environment", "도로": "traffic", "교통": "traffic",
        "시설": "facilities", "민원": "civil_service", "복지": "welfare", "기타": "other", "other": "other"
    }
    clean = re.sub(r'<thought>.*?</thought>', '', response, flags=re.DOTALL)
    if '</thought>' in response: clean = response.split('</thought>')[-1]
    clean = clean.strip().lower()
    for ko, en in ko_to_en.items():
        if ko in clean: return en
    return "unknown"

def main():
    # Initialize WandB
    wandb.init(
        project="exaone-civil-complaint",
        name=f"m3-optimization-eval-{datetime.now().strftime('%Y%m%d-%H%M')}",
        config={
            "model": "EXAONE-Deep-7.8B-Merged",
            "engine": "vLLM",
            "repetition_penalty": 1.1,
            "max_model_len": 8192,
            "stage": "M3_Optimization_Phase_1_2"
        }
    )

    print("=" * 60)
    print("M3 Optimization Evaluation (v3) with WandB")
    print("=" * 60)

    test_data = load_test_data(TEST_DATA_PATH, max_samples=50)
    print(f"Loaded {len(test_data)} samples")

    print("\nInitializing vLLM Engine...")
    llm = LLM(
        model=MODEL_DIR,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.65
    )

    tokenizer = llm.get_tokenizer()
    
    # 1. Generation Benchmark
    print("\n[1/3] Benchmarking Generation & Latency...")
    gen_prompts = []
    for item in test_data:
        messages = [
            {"role": "system", "content": "당신은 지자체 민원 담당 공무원을 돕는 AI 어시스턴트입니다."},
            {"role": "user", "content": f"{item['instruction']}\n\n{item['input'][:3000]}"}
        ]
        gen_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    sampling_params = SamplingParams(
        temperature=0.6, top_p=0.95, max_tokens=256, repetition_penalty=1.1,
        stop=["[|user|]", "[|system|]", "[|assistant|]"]
    )

    start_time = time.perf_counter()
    outputs = llm.generate(gen_prompts, sampling_params)
    avg_latency = (time.perf_counter() - start_time) / len(test_data)
    
    clean_gens = [re.sub(r'<thought>.*?</thought>', '', o.outputs[0].text, flags=re.DOTALL).strip() for o in outputs]
    clean_refs = [re.sub(r'<thought>.*?</thought>', '', item['output'], flags=re.DOTALL).strip() for item in test_data]

    # 2. Quality Metrics
    print("\n[2/3] Computing Quality Metrics...")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rouge_l = np.mean([scorer.score(r, g)['rougeL'].fmeasure * 100 for r, g in zip(clean_refs, clean_gens)])
    P, R, F1 = bert_score.score(clean_gens, clean_refs, lang="ko", verbose=False)
    avg_bert_f1 = F1.mean().item() * 100

    # 3. Classification
    print("\n[3/3] Evaluating Classification...")
    class_prompts = []
    for item in test_data:
        class_text = f"다음 민원을 한 단어로 분류하세요: [환경, 교통, 시설, 민원, 복지, 기타]\n\n민원: {item['input'][:1500]}\n\n결과:"
        messages = [
            {"role": "system", "content": "당신은 민원 분류기입니다. 반드시 한 단어로만 대답하세요."},
            {"role": "user", "content": class_text}
        ]
        class_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    class_outputs = llm.generate(class_prompts, SamplingParams(temperature=0.0, max_tokens=10, stop=["\n", " "]))
    
    correct = 0
    total = 0
    table_data = []
    for i, out in enumerate(class_outputs):
        true_cat = extract_true_category(test_data[i]['input'])
        if true_cat == "unknown": continue
        pred_cat = parse_predicted_category(out.outputs[0].text)
        is_correct = (pred_cat == true_cat)
        if is_correct: correct += 1
        total += 1
        if i < 10: table_data.append([true_cat, pred_cat, out.outputs[0].text[:30]])

    accuracy = (correct / total * 100) if total > 0 else 0

    # Final Log
    metrics = {
        "avg_latency": avg_latency,
        "rouge_l": rouge_l,
        "bert_score_f1": avg_bert_f1,
        "classification_accuracy": accuracy
    }
    wandb.log(metrics)
    
    print("\n" + "=" * 50)
    print(f"RESULTS: Latency {avg_latency:.3fs}, ROUGE-L {rouge_l:.2f}, BERT {avg_bert_f1:.2f}, Acc {accuracy:.2f}%")
    print("=" * 50)
    
    wandb.finish()

if __name__ == "__main__":
    main()
