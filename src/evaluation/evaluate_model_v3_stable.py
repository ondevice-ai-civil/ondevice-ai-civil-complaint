
import os
import sys
import time
import json
import torch
import numpy as np
import wandb
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import bert_score

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
            print(f"Applying final patch to {name}...")
            # Fix module-level constants
            class MockAttn:
                def get_interface(self, *args, **kwargs): return mod.eager_attention_forward
            mod.ALL_ATTENTION_FUNCTIONS = MockAttn()
            mod.auto_docstring = lambda *a, **k: (lambda x: x)
            mod.can_return_tuple = lambda x: x
            mod.dynamic_rope_update = lambda x: x
            mod.GradientCheckpointingLayer = torch.nn.Module
            
            # Fix Class methods
            if hasattr(mod, 'ExaoneModel'):
                mod.ExaoneModel.get_input_embeddings = lambda self: self.wte
                mod.ExaoneModel.set_input_embeddings = lambda self, val: setattr(self, 'wte', val)
            
            if hasattr(mod, 'ExaoneForCausalLM'):
                mod.ExaoneForCausalLM.get_input_embeddings = lambda self: self.transformer.wte
                mod.ExaoneForCausalLM.set_input_embeddings = lambda self, val: setattr(self.transformer, 'wte', val)
                
                # CRITICAL: Fix forward to prevent NoneType error
                orig_forward = mod.ExaoneForCausalLM.forward
                def patched_forward(self, *args, **kwargs):
                    if getattr(self, 'transformer', None) is None:
                        print("RE-INITIALIZING TRANSFORMER...")
                        self.transformer = mod.ExaoneModel(self.config)
                        # Sync device
                        self.transformer.to(self.device)
                    return orig_forward(self, *args, **kwargs)
                mod.ExaoneForCausalLM.forward = patched_forward

# Paths
BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
ADAPTER_ID = "umyunsang/civil-complaint-exaone-lora"
TEST_DATA_PATH = "/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_test.jsonl"

def main():
    wandb.init(project="exaone-civil-complaint", name=f"m3-final-stable-eval-{datetime.now().strftime('%Y%m%d-%H%M')}")

    print("Dynamic loading...")
    from transformers import AutoConfig
    AutoConfig.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    apply_final_runtime_patch()

    print("Loading model...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    
    apply_final_runtime_patch() # Post-load patch
    
    print("Loading Adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_ID)
    model.eval()

    # Minimal test
    test_data = []
    with open(TEST_DATA_PATH, 'r') as f:
        for line in f: test_data.append(json.loads(line))
    test_data = test_data[:3]

    print("Evaluating...")
    latencies = []
    clean_gens = []
    clean_refs = []
    
    for item in test_data:
        prompt = tokenizer.apply_chat_template([{"role":"user", "content":item['input'][:500]}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
        latencies.append(time.perf_counter()-start)
        gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Gen: {gen[:30]}...")
        clean_gens.append(gen.strip())
        clean_refs.append(item['output'].strip())

    P, R, F1 = bert_score.score(clean_gens, clean_refs, lang="ko")
    metrics = {"avg_latency": np.mean(latencies), "bert_score_f1": F1.mean().item() * 100}
    wandb.log(metrics)
    print(f"Metrics: {metrics}")
    wandb.finish()

if __name__ == "__main__":
    main()
