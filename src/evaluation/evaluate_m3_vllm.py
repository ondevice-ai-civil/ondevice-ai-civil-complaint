
import os
import sys
import time
import json
import re
import numpy as np
import wandb
from datetime import datetime
from vllm import LLM, SamplingParams
import bert_score
from rouge_score import rouge_scorer

# Constants
MODEL_ID = "umyunsang/civil-complaint-exaone-awq"
TEST_DATA_PATH = "/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_test.jsonl"

# 1. Environment fixes
import transformers.utils.generic
if not hasattr(transformers.utils.generic, 'check_model_inputs'):
    transformers.utils.generic.check_model_inputs = lambda *args, **kwargs: (args[1] if len(args) > 1 else kwargs.get('model_inputs'))

def load_test_data(path, max_samples=50):
    import random
    random.seed(42)
    data = []
    with open(path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except: continue
    random.shuffle(data)
    return data[:max_samples]

def extract_true_category(text):
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

def parse_m3_category(response, categories):
    # Phase 3: Robust parsing for EXAONE <thought> architecture
    clean = re.sub(r'<thought>.*?</thought>', '', response, flags=re.DOTALL)
    if '</thought>' in response:
        clean = response.split('</thought>')[-1]
    
    clean = clean.strip().lower()
    ko_to_en = {
        "환경": "environment", "교통": "traffic", "도로": "traffic", 
        "시설": "facilities", "민원": "civil_service", "복지": "welfare", "기타": "other"
    }
    for ko, en in ko_to_en.items():
        if ko in clean: return en
    for cat in categories:
        if cat in clean: return cat
    return "unknown"

def main():
    wandb.init(
        project="exaone-civil-complaint",
        name=f"m3-exaone-vllm-final-{datetime.now().strftime('%m%d-%H%M')}",
        config={
            "engine": "vLLM",
            "model": MODEL_ID,
            "repetition_penalty": 1.1,
            "quantization": "AWQ",
            "phase": "M3 Final Verification"
        }
    )

    print(f"Initializing vLLM with {MODEL_ID}...")
    # vLLM handles EXAONE architecture, but we need to ensure the dynamic code works
    try:
        llm = LLM(
            model=MODEL_ID,
            quantization="awq",
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.7, # Leave room for BERTScore
            dtype="float16"
        )
    except Exception as e:
        print(f"vLLM Initialization failed: {e}")
        # Fallback to HF if vLLM fails
        return

    test_data = load_test_data(TEST_DATA_PATH, max_samples=20)
    tokenizer = llm.get_tokenizer()

    print("\n[M3 Roadmap] Running Batch Evaluation...")
    
    gen_prompts = []
    for item in test_data:
        messages = [
            {"role": "system", "content": "당신은 민원 담당 공무원입니다."},
            {"role": "user", "content": f"{item['instruction']}\n\n{item['input'][:1000]}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        gen_prompts.append(prompt)

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=256,
        repetition_penalty=1.1, # Phase 1
        stop=["[|user|]", "[|system|]", "[|assistant|]", "[|endofturn|]"]
    )

    start_time = time.perf_counter()
    outputs = llm.generate(gen_prompts, sampling_params)
    avg_latency = (time.perf_counter() - start_time) / len(test_data)

    clean_gens = [re.sub(r'<thought>.*?</thought>', '', o.outputs[0].text, flags=re.DOTALL).strip() for o in outputs]
    clean_refs = [re.sub(r'<thought>.*?</thought>', '', item['output'], flags=re.DOTALL).strip() for item in test_data]

    print("\nCalculating M3 Metrics...")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rouge_l = np.mean([scorer.score(r, g)['rougeL'].fmeasure * 100 for r, g in zip(clean_refs, clean_gens)])
    
    # BERTScore
    P, R, F1 = bert_score.score(clean_gens, clean_refs, lang="ko", verbose=False)
    avg_bert_f1 = F1.mean().item() * 100

    # Phase 3: Classification
    correct, total = 0, 0
    categories = ["environment", "traffic", "facilities", "civil_service", "welfare", "other"]
    
    for i, out in enumerate(outputs):
        true_cat = extract_true_category(test_data[i]['input'])
        if true_cat == "unknown": continue
        pred_cat = parse_m3_category(out.outputs[0].text, categories)
        if pred_cat == true_cat: correct += 1
        total += 1

    accuracy = (correct / total * 100) if total > 0 else 0

    results = {
        "m3_latency_p50": avg_latency,
        "m3_rouge_l": rouge_l,
        "m3_bert_score_f1": avg_bert_f1,
        "m3_classification_accuracy": accuracy
    }
    wandb.log(results)

    print("\n" + "="*50)
    print("M3 FINAL OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Avg Latency: {avg_latency:.2f}s (KPI < 2s)")
    print(f"ROUGE-L: {rouge_l:.2f} (KPI >= 40)")
    print(f"BERTScore F1: {avg_bert_f1:.2f}")
    print(f"Classification Acc: {accuracy:.2f}% (KPI >= 85%)")
    print("="*50)
    
    wandb.finish()

if __name__ == "__main__":
    main()
