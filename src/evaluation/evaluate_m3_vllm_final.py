
import os
import sys
import time
import json
import re
import numpy as np
import wandb
from datetime import datetime

# 1. Critical Runtime Library Patching
import transformers.modeling_rope_utils
if not hasattr(transformers.modeling_rope_utils, 'RopeParameters'):
    class RopeParameters(dict): pass
    transformers.modeling_rope_utils.RopeParameters = RopeParameters

import transformers.utils.generic
if not hasattr(transformers.utils.generic, 'check_model_inputs'):
    transformers.utils.generic.check_model_inputs = lambda *args, **kwargs: None

# 2. VLLM Initialization
from vllm import LLM, SamplingParams
import bert_score
from rouge_score import rouge_scorer

MODEL_DIR = '/content/civil-complaint-exaone-awq'
TEST_DATA_PATH = "/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_test.jsonl"

def main():
    wandb.init(
        project="exaone-civil-complaint",
        name=f"m3-exaone-vllm-final-success-{datetime.now().strftime('%m%d-%H%M')}",
        config={"engine": "vLLM", "roadmap": "M3 Phase 1,2,3", "repetition_penalty": 1.1}
    )

    print("Initializing Optimized vLLM Engine...")
    llm = LLM(
        model=MODEL_DIR, # Use our patched local config
        tokenizer=MODEL_DIR,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.75,
        dtype="float16",
        quantization="awq" # Specifically tell vLLM it is AWQ
    )

    # Data loading
    test_data = []
    with open(TEST_DATA_PATH, 'r') as f:
        for line in f: test_data.append(json.loads(line))
    test_data = test_data[:20]

    tokenizer = llm.get_tokenizer()
    
    # Phase 1 & 2: Benchmarking
    prompts = []
    for item in test_data:
        messages = [{"role": "user", "content": f"{item['instruction']}\n\n{item['input'][:1000]}"}]
        p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(p)

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=256,
        repetition_penalty=1.1, # Phase 1
        stop=["[|user|]", "[|system|]", "[|assistant|]"]
    )

    print("Running Inference...")
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    avg_latency = (time.perf_counter() - start) / len(test_data)

    clean_gens = [re.sub(r'<thought>.*?</thought>', '', o.outputs[0].text, flags=re.DOTALL).strip() for o in outputs]
    clean_refs = [re.sub(r'<thought>.*?</thought>', '', item['output'], flags=re.DOTALL).strip() for item in test_data]

    # Metrics
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rouge_l = np.mean([scorer.score(r, g)['rougeL'].fmeasure * 100 for r, g in zip(clean_refs, clean_gens)])
    P, R, F1 = bert_score.score(clean_gens, clean_refs, lang="ko")
    
    # Phase 3: Classification logic
    correct, total = 0, 0
    categories = ["environment", "traffic", "facilities", "civil_service", "welfare", "other"]
    for i, out in enumerate(outputs):
        raw = out.outputs[0].text.lower()
        pred = "unknown"
        ko_map = {"환경": "environment", "교통": "traffic", "시설": "facilities", "민원": "civil_service", "복지": "welfare"}
        for k, v in ko_map.items(): 
            if k in raw: pred = v
        
        # Simple extraction for demo
        match = re.search(r'\[Category:\s*([^\]]+)\]', test_data[i]['input'])
        true_cat = match.group(1).strip().lower() if match else "other"
        if pred == true_cat or (pred == "unknown" and true_cat == "other"): correct += 1
        total += 1

    metrics = {
        "m3_avg_latency": avg_latency,
        "m3_rouge_l": rouge_l,
        "m3_bert_score_f1": F1.mean().item() * 100,
        "m3_classification_acc": (correct / total) * 100
    }
    wandb.log(metrics)
    print(f"\nFINAL M3 METRICS: {metrics}")
    wandb.finish()

if __name__ == "__main__":
    main()
