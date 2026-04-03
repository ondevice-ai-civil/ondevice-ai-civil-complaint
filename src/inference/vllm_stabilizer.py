"""
vLLM Stabilizer for EXAONE Model

Initializes and tests vLLM engine for EXAONE-based civil complaint model.
Note: As of transformers 4.53.0+, RopeParameters, check_model_inputs,
and ALL_ATTENTION_FUNCTIONS exist natively — no monkey-patches needed.
"""

import sys

import torch
from loguru import logger


def apply_transformers_patch():
    """No-op: transformers 4.53.0+ includes all previously patched APIs natively.

    Retained for backward compatibility with api_server.py and test mocks.
    """
    logger.debug("apply_transformers_patch called (no-op for transformers 4.53.0+)")


def start_vllm_engine(model_id):
    """Initialize vLLM engine with conservative settings for EXAONE AWQ model."""
    from vllm import LLM, SamplingParams  # noqa: F401

    logger.info(f"Initializing vLLM Engine with model: {model_id}")

    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.8,
        dtype="float16",
        enforce_eager=True,
    )

    return llm


def test_exaone_generation(engine):
    """Run a quick sanity check on the loaded vLLM engine."""
    from vllm import SamplingParams

    prompts = ["당신은 민원 전문가입니다. 간단히 인사하세요."]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
    outputs = engine.generate(prompts, sampling_params)

    for output in outputs:
        logger.info(f"Sanity Check Output: {output.outputs[0].text}")

    return outputs


if __name__ == "__main__":
    MODEL_ID = "umyunsang/civil-complaint-exaone-awq"

    try:
        engine = start_vllm_engine(MODEL_ID)
        logger.info("=" * 50)
        logger.info("VLLM ENGINE STABILIZED AND LOADED SUCCESSFULLY!")
        logger.info("=" * 50)

        test_exaone_generation(engine)

    except Exception as e:
        logger.error(f"Stabilization failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
