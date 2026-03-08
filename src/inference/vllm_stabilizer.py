
import os
import sys
import torch

# 1. Critical Runtime Patch for EXAONE Compatibility
def apply_transformers_patch():
    print("Applying runtime patches for EXAONE...")
    
    # Patch 1: RopeParameters injection
    import transformers.modeling_rope_utils
    if not hasattr(transformers.modeling_rope_utils, 'RopeParameters'):
        # Define a dummy class that behaves like a dict (which RopeParameters is in newer versions)
        class RopeParameters(dict):
            pass
        transformers.modeling_rope_utils.RopeParameters = RopeParameters
        print("  [SUCCESS] Injected RopeParameters into transformers.modeling_rope_utils")

    # Patch 2: check_model_inputs injection
    import transformers.utils.generic
    if not hasattr(transformers.utils.generic, 'check_model_inputs'):
        transformers.utils.generic.check_model_inputs = lambda *args, **kwargs: None
        print("  [SUCCESS] Injected check_model_inputs into transformers.utils.generic")

    # Patch 3: ALL_ATTENTION_FUNCTIONS (Required for some versions)
    try:
        import transformers.modeling_utils
        if not hasattr(transformers.modeling_utils, 'ALL_ATTENTION_FUNCTIONS'):
            class MockAttn:
                def get(self, name, default=None): return None
            transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS = MockAttn()
            print("  [SUCCESS] Injected ALL_ATTENTION_FUNCTIONS dummy")
    except:
        pass

# 2. VLLM Initialization
def start_vllm_engine(model_id):
    from vllm import LLM, SamplingParams
    
    print(f"Initializing vLLM Engine with model: {model_id}")
    
    # We use a more conservative memory setting to ensure stability
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.8,
        dtype="float16", # EXAONE-Deep-AWQ is float16/bfloat16
        enforce_eager=True # Eager mode is often more stable for patched models
    )
    
    return llm

if __name__ == "__main__":
    apply_transformers_patch()
    
    # Test with the already uploaded AWQ model
    MODEL_ID = "umyunsang/civil-complaint-exaone-awq"
    
    try:
        engine = start_vllm_engine(MODEL_ID)
        print("\n" + "="*50)
        print("VLLM ENGINE STABILIZED AND LOADED SUCCESSFULLY!")
        print("="*50)
        
        # Quick Sanity Check
        prompts = ["당신은 민원 전문가입니다. 간단히 인사하세요."]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
        outputs = engine.generate(prompts, sampling_params)
        
        for output in outputs:
            print(f"Sanity Check Output: {output.outputs[0].text}")
            
    except Exception as e:
        print(f"\n[ERROR] Stabilization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
