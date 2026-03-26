import os
import sys
import time
import json
import re
import numpy as np
import wandb
import torch
from datetime import datetime
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import bert_score
from rouge_score import rouge_scorer

# Constants
MODEL_ID = "./final_model"
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
    clean = re.sub(r"<thought>.*?</thought>", "", response, flags=re.DOTALL)
    if "</thought>" in response:
        clean = response.split("</thought>")[-1]
    clean = clean.strip().lower()
    for ko, en in ko_to_en.items():
        if ko in clean:
            return en
    return "unknown"


def main():
    wandb.init(
        project="exaone-civil-complaint",
        name=f"m3-exaone-awq-final-{datetime.now().strftime('%m%d-%H%M')}",
        config={"phase": "M3 Roadmap Final Stable", "model": MODEL_ID, "engine": "AutoAWQ"},
    )

    print(f"Loading {MODEL_ID} with AutoAWQ...")
    # Manually load the tokenizer to bypass AutoTokenizer errors
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained("./patched_model", trust_remote_code=True)
    # Ensure chat template is set if missing
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{'[|' + message['role'] + '|]' + message['content']}}{% if not loop.last %}{{ '[|endofturn|]\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '[|assistant|]<thought>\n' }}{% endif %}"

    model = AutoAWQForCausalLM.from_quantized(
        MODEL_ID, fuse_layers=True, trust_remote_code=True, safetensors=True
    )

    test_data = load_test_data(TEST_DATA_PATH, max_samples=10)

    print("Running M3 Roadmap Evaluation...")
    latencies, clean_gens, clean_refs, correct = [], [], [], 0

    for i, item in enumerate(test_data):
        messages = [
            {"role": "system", "content": "당신은 민원 공무원입니다."},
            {"role": "user", "content": f"{item['instruction']}\n\n{item['input'][:1000]}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        # Fix: EXAONE does not use token_type_ids
        inputs.pop("token_type_ids", None)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, repetition_penalty=1.1)
        latencies.append(time.perf_counter() - start)

        gen_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        clean_gens.append(re.sub(r"<thought>.*?</thought>", "", gen_text, flags=re.DOTALL).strip())
        clean_refs.append(
            re.sub(r"<thought>.*?</thought>", "", item["output"], flags=re.DOTALL).strip()
        )

        print(f"Sample {i+1} Gen: {clean_gens[-1][:50]}...")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l = np.mean(
        [scorer.score(r, g)["rougeL"].fmeasure * 100 for r, g in zip(clean_refs, clean_gens)]
    )
    P, R, F1 = bert_score.score(clean_gens, clean_refs, lang="ko")

    res = {
        "avg_latency": np.mean(latencies),
        "rouge_l": rouge_l,
        "bert_score_f1": F1.mean().item() * 100,
    }
    wandb.log(res)
    print(f"\nFinal M3 Results: {res}")
    wandb.finish()


if __name__ == "__main__":
    main()
