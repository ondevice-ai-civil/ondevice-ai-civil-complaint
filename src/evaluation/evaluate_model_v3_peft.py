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
from peft import PeftModel
import bert_score
from rouge_score import rouge_scorer

# 1. Structural Fixes for library compatibility
try:
    import transformers.utils.auto_docstring

    transformers.utils.auto_docstring.auto_docstring = lambda *args, **kwargs: (lambda obj: obj)
except ImportError:
    pass

import transformers.utils.generic
import transformers.modeling_rope_utils
import transformers.integrations

if not hasattr(transformers.utils.generic, "check_model_inputs"):
    transformers.utils.generic.check_model_inputs = lambda *args, **kwargs: (
        args[1] if len(args) > 1 else kwargs.get("model_inputs")
    )
if not hasattr(transformers.utils.generic, "maybe_autocast"):
    from contextlib import nullcontext

    transformers.utils.generic.maybe_autocast = lambda *args, **kwargs: nullcontext()
if not hasattr(transformers.modeling_rope_utils, "RopeParameters"):

    class RopeParameters(dict):
        pass

    transformers.modeling_rope_utils.RopeParameters = RopeParameters


# 2. Advanced Runtime Patching
def apply_exaone_structural_patch():
    try:
        # We need to find the dynamic modules
        for name, mod in sys.modules.items():
            if "modeling_exaone" in name:
                # Fix missing imports in module
                if not hasattr(mod, "ALL_ATTENTION_FUNCTIONS"):

                    class Dummy:
                        pass

                    mod.ALL_ATTENTION_FUNCTIONS = Dummy()
                    mod.ALL_ATTENTION_FUNCTIONS.get_interface = (
                        lambda *a, **k: mod.eager_attention_forward
                    )

                if not hasattr(mod, "GradientCheckpointingLayer"):
                    mod.GradientCheckpointingLayer = torch.nn.Module

                if not hasattr(mod, "dynamic_rope_update"):
                    mod.dynamic_rope_update = lambda x: x

                if not hasattr(mod, "auto_docstring"):
                    mod.auto_docstring = lambda *a, **k: (lambda x: x)

                if not hasattr(mod, "can_return_tuple"):
                    mod.can_return_tuple = lambda x: x

                # Fix missing attributes in classes
                if hasattr(mod, "ExaoneModel"):
                    if not hasattr(mod.ExaoneModel, "get_input_embeddings"):
                        mod.ExaoneModel.get_input_embeddings = lambda self: self.wte
                        mod.ExaoneModel.set_input_embeddings = lambda self, value: setattr(
                            self, "wte", value
                        )

                if hasattr(mod, "ExaoneForCausalLM"):
                    if not hasattr(mod.ExaoneForCausalLM, "get_input_embeddings"):
                        mod.ExaoneForCausalLM.get_input_embeddings = (
                            lambda self: self.transformer.wte
                        )
                        mod.ExaoneForCausalLM.set_input_embeddings = lambda self, value: setattr(
                            self.transformer, "wte", value
                        )
    except Exception as e:
        print(f"Structural patch failed: {e}")


# Paths
BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
ADAPTER_ID = "umyunsang/civil-complaint-exaone-lora"
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
        return mapping.get(cat, "other")
    return "other"


def parse_predicted_category(response):
    ko_to_en = {
        "환경": "environment",
        "도로": "traffic",
        "시설": "facilities",
        "민원": "civil_service",
        "복지": "welfare",
        "기타": "other",
    }
    clean = re.sub(r"<thought>.*?</thought>", "", response, flags=re.DOTALL)
    if "</thought>" in response:
        clean = response.split("</thought>")[-1]
    for ko, en in ko_to_en.items():
        if ko in clean:
            return en
    return "unknown"


def main():
    wandb.init(
        project="exaone-civil-complaint",
        name=f"m3-peft-final-direct-{datetime.now().strftime('%Y%m%d-%H%M')}",
    )

    print("Loading Base Model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Trigger dynamic import
    from transformers import AutoConfig

    AutoConfig.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    # Patch now
    apply_exaone_structural_patch()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )

    # Re-apply patch after model load just in case
    apply_exaone_structural_patch()

    print("Loading LoRA Adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    model.eval()

    test_data = load_test_data(TEST_DATA_PATH, max_samples=20)

    print("Running Generation & Classification...")
    latencies = []
    clean_gens = []
    clean_refs = []
    correct = 0

    for i, item in enumerate(test_data):
        messages = [
            {"role": "system", "content": "당신은 민원 담당 AI입니다. 한국어로 답변하세요."},
            {"role": "user", "content": f"{item['instruction']}\n\n{item['input'][:1000]}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        cg = re.sub(r"<thought>.*?</thought>", "", gen_text, flags=re.DOTALL).strip()
        cr = re.sub(r"<thought>.*?</thought>", "", item["output"], flags=re.DOTALL).strip()
        clean_gens.append(cg)
        clean_refs.append(cr)

        class_prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "민원 분류기입니다. 반드시 한 단어(환경, 교통, 시설, 민원, 복지, 기타)로만 답하세요.",
                },
                {"role": "user", "content": f"민원: {item['input'][:500]}\n결과:"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        class_inputs = tokenizer(class_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            class_outputs = model.generate(**class_inputs, max_new_tokens=10)
        class_text = tokenizer.decode(
            class_outputs[0][class_inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        true_cat = extract_true_category(item["input"])
        pred_cat = parse_predicted_category(class_text)
        if pred_cat == true_cat:
            correct += 1

    # Metrics
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l = np.mean(
        [scorer.score(r, g)["rougeL"].fmeasure * 100 for r, g in zip(clean_refs, clean_gens)]
    )
    P, R, F1 = bert_score.score(clean_gens, clean_refs, lang="ko")

    metrics = {
        "avg_latency": np.mean(latencies),
        "rouge_l": rouge_l,
        "bert_score_f1": F1.mean().item() * 100,
        "classification_accuracy": (correct / len(test_data)) * 100,
    }
    wandb.log(metrics)
    print(f"\nFinal Metrics: {metrics}")
    wandb.finish()


if __name__ == "__main__":
    main()
