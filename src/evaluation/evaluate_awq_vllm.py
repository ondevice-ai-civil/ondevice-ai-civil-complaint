import os
import sys
import time
import json
import re
import numpy as np
from datetime import datetime
from loguru import logger

# 1. Critical Runtime Library Patching for EXAONE
import transformers.modeling_rope_utils
if not hasattr(transformers.modeling_rope_utils, 'RopeParameters'):
    class RopeParameters(dict): pass
    transformers.modeling_rope_utils.RopeParameters = RopeParameters

import transformers.utils.generic
if not hasattr(transformers.utils.generic, 'check_model_inputs'):
    transformers.utils.generic.check_model_inputs = lambda *args, **kwargs: None

# 2. Imports
try:
    from vllm import LLM, SamplingParams
except ImportError:
    logger.error("vLLM is not installed. Please install it with 'pip install vllm'.")
    sys.exit(1)

try:
    import bert_score
    from rouge_score import rouge_scorer
except ImportError:
    logger.warning("bert_score or rouge_score not found. Some metrics will be skipped.")

# --- Configuration ---
MODEL_ID = "umyunsang/civil-complaint-exaone-awq"
TEST_DATA_PATH = "GovOn/data/processed/v2_test.jsonl"
NUM_TEST_SAMPLES = 50  # Balanced sample size for quick validation

def main():
    logger.info(f"Starting AWQ Model Evaluation: {MODEL_ID}")
    
    # Initialize vLLM
    try:
        llm = LLM(
            model=MODEL_ID,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.8,
            dtype="float16",
            quantization="awq",
            enforce_eager=True
        )
    except Exception as e:
        logger.error(f"Failed to initialize vLLM: {e}")
        return

    # Data loading
    test_data = []
    if not os.path.exists(TEST_DATA_PATH):
        logger.error(f"Test data not found at {TEST_DATA_PATH}")
        return

    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    # Shuffle and sample
    indices = np.random.choice(len(test_data), min(NUM_TEST_SAMPLES, len(test_data)), replace=False)
    test_samples = [test_data[i] for i in indices]
    
    tokenizer = llm.get_tokenizer()
    prompts = []
    for item in test_samples:
        # Extract prompt from 'text' field if it contains the chat template
        raw_text = item.get('text', "")
        if "[|user|]" in raw_text:
            user_prompt = raw_text.split("[|user|]")[1].split("[|endofturn|]")[0].strip()
            prompts.append(f"[|system|]당신은 지자체 민원 담당 공무원을 돕는 AI 어시스턴트 입니다.[|endofturn|][|user|]{user_prompt}[|assistant|]")
        else:
            prompts.append(f"[|user|]{item.get('input', '')}[|assistant|]")

    sampling_params = SamplingParams(
        temperature=0.0, # Greedy search for deterministic results
        max_tokens=512,
        repetition_penalty=1.1,
        stop=["[|user|]", "[|system|]", "[|assistant|]"]
    )

    logger.info(f"Running inference on {len(prompts)} samples...")
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.perf_counter() - start_time
    
    avg_latency = total_time / len(test_samples)
    
    # Process outputs
    clean_gens = [o.outputs[0].text.strip() for o in outputs]
    
    # Extract reference output (strip <thought> if present)
    clean_refs = []
    for item in test_samples:
        ref = item.get('output', '')
        if not ref and 'text' in item:
            # Fallback to parsing 'text' field
            if "[|assistant|]" in item['text']:
                ref = item['text'].split("[|assistant|]")[1].split("[|endofturn|]")[0].strip()
        
        # Remove <thought> tags from reference
        ref = re.sub(r'<thought>.*?</thought>', '', ref, flags=re.DOTALL).strip()
        clean_refs.append(ref)

    # Metrics calculation
    metrics = {"avg_latency_sec": avg_latency}
    
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        rouge_l = np.mean([scorer.score(r, g)['rougeL'].fmeasure * 100 for r, g in zip(clean_refs, clean_gens)])
        metrics["rouge_l"] = rouge_l
        
        P, R, F1 = bert_score.score(clean_gens, clean_refs, lang="ko", verbose=False)
        metrics["bert_score_f1"] = F1.mean().item() * 100
    except Exception as e:
        logger.warning(f"Error calculating ROUGE/BERTScore: {e}")

    logger.info("\n" + "="*50)
    logger.info(f"EVALUATION RESULTS - {MODEL_ID}")
    logger.info("="*50)
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info("="*50)
    
    # Save results
    output_file = f"evaluation_awq_{datetime.now().strftime('%m%d_%H%M')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"metrics": metrics, "samples": [{"ref": r, "gen": g} for r, g in zip(clean_refs, clean_gens)]}, f, ensure_ascii=False, indent=2)
    logger.info(f"Detailed results saved to {output_file}")

if __name__ == "__main__":
    main()
