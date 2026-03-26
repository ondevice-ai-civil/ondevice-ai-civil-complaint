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


# CRITICAL: Structural patches at runtime instead of editing files
def patch_exaone_dynamic():
    import transformers.utils.generic
    import transformers.modeling_rope_utils

    # Fix 1: check_model_inputs
    if not hasattr(transformers.utils.generic, "check_model_inputs"):
        transformers.utils.generic.check_model_inputs = lambda *args, **kwargs: (
            args[1] if len(args) > 1 else kwargs.get("model_inputs")
        )
    # Fix 2: RopeParameters
    if not hasattr(transformers.modeling_rope_utils, "RopeParameters"):

        class RopeParameters(dict):
            pass

        transformers.modeling_rope_utils.RopeParameters = RopeParameters


# Paths
MODEL_ID = "LGAI-EXAONE/EXAONE-Deep-7.8B"
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
    clean = re.sub(r"<thought>.*?</thought>", "", response, flags=re.DOTALL)
    if "</thought>" in response:
        clean = response.split("</thought>")[-1]
    clean = clean.strip().lower()
    for ko, en in ko_to_en.items():
        if ko in clean:
            return en
    return "unknown"


def main():
    patch_exaone_dynamic()

    wandb.init(
        project="exaone-civil-complaint",
        name=f"m3-exaone-final-verified-{datetime.now().strftime('%m%d-%H%M')}",
        config={
            "phase": "M3 Roadmap Final",
            "model": "EXAONE-Deep-7.8B",
            "repetition_penalty": 1.1,
        },
    )

    print("Loading Model (HF 4-bit)...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Trigger dynamic load
    from transformers import AutoConfig

    AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Patch module after load
    for name, mod in sys.modules.items():
        if "modeling_exaone" in name:
            if not hasattr(mod, "ALL_ATTENTION_FUNCTIONS"):

                class Mock:
                    def get_interface(self, *a, **k):
                        return mod.eager_attention_forward

                mod.ALL_ATTENTION_FUNCTIONS = Mock()
            mod.auto_docstring = lambda *a, **k: (lambda x: x)
            mod.can_return_tuple = lambda x: x
            mod.dynamic_rope_update = lambda x: x
            mod.GradientCheckpointingLayer = torch.nn.Module

            # Methods
            mod.ExaoneModel.get_input_embeddings = lambda self: self.wte
            mod.ExaoneModel.set_input_embeddings = lambda self, v: setattr(self, "wte", v)
            mod.ExaoneForCausalLM.get_input_embeddings = lambda self: self.transformer.wte
            mod.ExaoneForCausalLM.set_input_embeddings = lambda self, v: setattr(
                self.transformer, "wte", v
            )

            # prepare_inputs
            def pi(
                self,
                input_ids,
                past_key_values=None,
                attention_mask=None,
                inputs_embeds=None,
                **kwargs,
            ):
                if past_key_values is not None:
                    past_length = (
                        past_key_values.get_seq_length()
                        if hasattr(past_key_values, "get_seq_length")
                        else past_key_values[0][0].shape[2]
                    )
                    if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                        input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
                    elif past_length <= input_ids.shape[1]:
                        input_ids = input_ids[:, past_length:]
                model_inputs = {"input_ids": input_ids}
                model_inputs.update(
                    {
                        "past_key_values": past_key_values,
                        "use_cache": kwargs.get("use_cache"),
                        "attention_mask": attention_mask,
                    }
                )
                return model_inputs

            mod.ExaoneForCausalLM.prepare_inputs_for_generation = pi

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    model.eval()

    test_data = load_test_data(TEST_DATA_PATH, max_samples=20)

    print("Running Evaluation...")
    latencies, clean_gens, clean_refs, correct = [], [], [], 0

    for i, item in enumerate(test_data):
        messages = [
            {"role": "system", "content": "당신은 민원 공무원입니다."},
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
        clean_gens.append(re.sub(r"<thought>.*?</thought>", "", gen_text, flags=re.DOTALL).strip())
        clean_refs.append(
            re.sub(r"<thought>.*?</thought>", "", item["output"], flags=re.DOTALL).strip()
        )

        if parse_m3_category(gen_text) == extract_true_category(item["input"]):
            correct += 1

    # Metrics
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
    print(f"\nFinal M3 Results: {res}")
    wandb.finish()


if __name__ == "__main__":
    main()
