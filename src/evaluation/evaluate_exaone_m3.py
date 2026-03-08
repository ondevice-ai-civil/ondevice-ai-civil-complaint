
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

# 1. Structural Fixes
import transformers.utils.generic
import transformers.modeling_rope_utils
import transformers.integrations

if not hasattr(transformers.utils.generic, 'check_model_inputs'):
    transformers.utils.generic.check_model_inputs = lambda *args, **kwargs: (args[1] if len(args) > 1 else kwargs.get('model_inputs'))
if not hasattr(transformers.utils.generic, 'maybe_autocast'):
    from contextlib import nullcontext
    transformers.utils.generic.maybe_autocast = lambda *args, **kwargs: nullcontext()
if not hasattr(transformers.modeling_rope_utils, 'RopeParameters'):
    class RopeParameters(dict): pass
    transformers.modeling_rope_utils.RopeParameters = RopeParameters

def apply_final_runtime_patch():
    for name, mod in sys.modules.items():
        if 'modeling_exaone' in name:
            print(f"Patching {name}...")
            
            # ALL_ATTENTION_FUNCTIONS mock
            if not hasattr(mod, 'ALL_ATTENTION_FUNCTIONS'):
                class Mock:
                    def get_interface(self, *a, **k): return mod.eager_attention_forward
                mod.ALL_ATTENTION_FUNCTIONS = Mock()
            
            # Class methods
            if hasattr(mod, 'ExaoneModel'):
                mod.ExaoneModel.get_input_embeddings = lambda self: self.wte
                mod.ExaoneModel.set_input_embeddings = lambda self, v: setattr(self, 'wte', v)
            
            if hasattr(mod, 'ExaoneForCausalLM'):
                mod.ExaoneForCausalLM.get_input_embeddings = lambda self: self.transformer.wte
                mod.ExaoneForCausalLM.set_input_embeddings = lambda self, v: setattr(self.transformer, 'wte', v)
                
                # Fix the NoneType forward
                orig_forward = mod.ExaoneForCausalLM.forward
                def patched_forward(self, *args, **kwargs):
                    if getattr(self, 'transformer', None) is None:
                        self.transformer = mod.ExaoneModel(self.config).to(self.device)
                    return orig_forward(self, *args, **kwargs)
                mod.ExaoneForCausalLM.forward = patched_forward
                
                # Fix prepare_inputs
                def prepare_inputs(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
                    if past_key_values is not None:
                        past_length = past_key_values[0][0].shape[2]
                        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
                        elif past_length <= input_ids.shape[1]:
                            input_ids = input_ids[:, past_length:]
                    model_inputs = {"input_ids": input_ids} if inputs_embeds is None or past_key_values is not None else {"inputs_embeds": inputs_embeds}
                    model_inputs.update({"past_key_values": past_key_values, "use_cache": kwargs.get("use_cache"), "attention_mask": attention_mask})
                    return model_inputs
                mod.ExaoneForCausalLM.prepare_inputs_for_generation = prepare_inputs

            mod.auto_docstring = lambda *a, **k: (lambda x: x)
            mod.can_return_tuple = lambda x: x
            mod.dynamic_rope_update = lambda x: x
            mod.GradientCheckpointingLayer = torch.nn.Module

# Paths
BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-Deep-7.8B"
ADAPTER_ID = "umyunsang/civil-complaint-exaone-lora"
TEST_DATA_PATH = "/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_test.jsonl"

def main():
    wandb.init(project="exaone-civil-complaint", name=f"m3-exaone-final-verified-{datetime.now().strftime('%m%d-%H%M')}")

    print("Loading model...")
    from transformers import AutoConfig
    AutoConfig.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    apply_final_runtime_patch()

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    
    apply_final_runtime_patch()
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    model.eval()

    # Data
    test_data = []
    with open(TEST_DATA_PATH, 'r') as f:
        for line in f: test_data.append(json.loads(line))
    test_data = test_data[:5] 

    print("Evaluating...")
    latencies, gens, refs = [], [], []
    for item in test_data:
        prompt = tokenizer.apply_chat_template([{"role":"user", "content":item['input'][:500]}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
        latencies.append(time.perf_counter()-start)
        gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Gen: {gen[:50]}...")
        gens.append(re.sub(r'<thought>.*?</thought>', '', gen, flags=re.DOTALL).strip())
        refs.append(re.sub(r'<thought>.*?</thought>', '', item['output'], flags=re.DOTALL).strip())

    P, R, F1 = bert_score.score(gens, refs, lang="ko")
    m = {"avg_latency": np.mean(latencies), "bert_score_f1": F1.mean().item() * 100}
    wandb.log(m)
    print(f"Final M3 Result: {m}")
    wandb.finish()

if __name__ == "__main__":
    main()
