
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bert_score
from rouge_score import rouge_scorer

# Structural Fixes for library compatibility
import transformers.utils.auto_docstring
transformers.utils.auto_docstring.auto_docstring = lambda *args, **kwargs: (lambda obj: obj)

import transformers.utils.generic
import transformers.modeling_rope_utils
import transformers.integrations
import transformers.masking_utils

if not hasattr(transformers.utils.generic, 'check_model_inputs'):
    transformers.utils.generic.check_model_inputs = lambda *args, **kwargs: (args[1] if len(args) > 1 else kwargs.get('model_inputs'))
if not hasattr(transformers.utils.generic, 'maybe_autocast'):
    from contextlib import nullcontext
    transformers.utils.generic.maybe_autocast = lambda *args, **kwargs: nullcontext()
if not hasattr(transformers.modeling_rope_utils, 'RopeParameters'):
    class RopeParameters(dict): pass
    transformers.modeling_rope_utils.RopeParameters = RopeParameters

# Paths
MODEL_DIR = "/content/dummy_model"
TEST_DATA_PATH = "/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_test.jsonl"

def load_test_data(path, max_samples=20):
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
        mapping = {"환경/위생": "environment", "도로/교통": "traffic", "시설물관리": "facilities", "민원서비스": "civil_service", "복지": "welfare", "기타": "other"}
        return mapping.get(cat, "other")
    return "other"

def parse_predicted_category(response):
    ko_to_en = {"환경": "environment", "도로": "traffic", "시설": "facilities", "민원": "civil_service", "복지": "welfare", "기타": "other"}
    clean = re.sub(r'<thought>.*?</thought>', '', response, flags=re.DOTALL)
    if '</thought>' in response: clean = response.split('</thought>')[-1]
    for ko, en in ko_to_en.items():
        if ko in clean: return en
    return "unknown"

def main():
    wandb.init(project="exaone-civil-complaint", name=f"m3-hf-final-dummy-{datetime.now().strftime('%Y%m%d-%H%M')}")

    print("Loading model (HF 4-bit, local dummy patched files)...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)

    test_data = load_test_data(TEST_DATA_PATH, max_samples=20)
    
    print("Running Generation & Classification...")
    latencies = []
    clean_gens = []
    clean_refs = []
    correct = 0
    
    for i, item in enumerate(test_data):
        # 1. Generation
        messages = [{"role": "system", "content": "당신은 민원 담당 AI입니다."}, {"role": "user", "content": f"{item['instruction']}\n\n{item['input'][:1000]}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, repetition_penalty=1.1, do_sample=True, temperature=0.6)
        latencies.append(time.perf_counter() - start)
        
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        cg = re.sub(r'<thought>.*?</thought>', '', gen_text, flags=re.DOTALL).strip()
        cr = re.sub(r'<thought>.*?</thought>', '', item['output'], flags=re.DOTALL).strip()
        clean_gens.append(cg)
        clean_refs.append(cr)
        
        # 2. Classification
        class_prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": "민원 분류기입니다. 한 단어(환경, 교통, 시설, 민원, 복지, 기타)로만 답하세요."},
            {"role": "user", "content": f"민원: {item['input'][:500]}\n결과:"}
        ], tokenize=False, add_generation_prompt=True)
        class_inputs = tokenizer(class_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            class_outputs = model.generate(**class_inputs, max_new_tokens=10)
        class_text = tokenizer.decode(class_outputs[0][class_inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        true_cat = extract_true_category(item['input'])
        pred_cat = parse_predicted_category(class_text)
        if pred_cat == true_cat: correct += 1

        if i < 5:
            print(f"\n--- Sample {i+1} ---")
            print(f"True Cat: {true_cat} | Pred Cat: {pred_cat} | Raw Class: {class_text.strip()}")
            print(f"Ref: {cr[:50]}...")
            print(f"Gen: {cg[:50]}...")

    # Metrics
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rouge_l = np.mean([scorer.score(r, g)['rougeL'].fmeasure * 100 for r, g in zip(clean_refs, clean_gens)])
    P, R, F1 = bert_score.score(clean_gens, clean_refs, lang="ko")
    
    metrics = {
        "avg_latency": np.mean(latencies),
        "rouge_l": rouge_l,
        "bert_score_f1": F1.mean().item() * 100,
        "classification_accuracy": (correct / len(test_data)) * 100
    }
    wandb.log(metrics)
    print(f"Final Metrics: {metrics}")
    wandb.finish()

if __name__ == "__main__":
    main()
