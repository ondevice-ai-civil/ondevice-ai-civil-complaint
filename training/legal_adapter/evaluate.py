"""
GovOn Legal Adapter 평가 스크립트
====================================
베이스 모델 대비 법령 조항 인용 비율, BERTScore, ROUGE-L 측정.

Usage:
    # vLLM 엔드포인트로 평가 (권장)
    OPENAI_API_BASE=http://localhost:8000/v1 python evaluate.py

    # 샘플 수 조정
    python evaluate.py --sample-size 50

Requirements:
    pip install datasets bert-score rouge-score openai tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_REPO = "umyunsang/govon-legal-response-data"
SYSTEM_PROMPT = (
    "당신은 대한민국 법률 전문가입니다. "
    "법령 조항과 판례를 정확하게 인용하여 법률 질문에 답변해 주세요."
)

# 법령 조항 인용 패턴
CITATION_PATTERNS = [
    r"제\s*\d+조",  # 제1조, 제 1 조
    r"법률\s*제\s*\d+호",  # 법률 제1234호
    r"제\s*\d+항",  # 제1항
    r"제\s*\d+호",  # 제1호
    r"\d{4}\s*년\s*법",  # 2024년 법
    r"판례\s*\d+",  # 판례 번호
    r"\d+가합\d+",  # 판결문 번호 형식
    r"\d+다\d+",  # 대법원 판결 번호
]
CITATION_REGEX = re.compile("|".join(CITATION_PATTERNS))


def has_citation(text: str) -> bool:
    return bool(CITATION_REGEX.search(text))


def generate_response(client, model: str, question: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=512,
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def evaluate(
    sample_size: int = 100,
    base_model: str = "LGAI-EXAONE/EXAONE-4.0-32B-AWQ",
    adapter_model: str = "legal-adapter",
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "dummy",
) -> Dict:
    from datasets import load_dataset
    from openai import OpenAI

    logger.info("Loading dataset: %s", DATASET_REPO)
    ds = load_dataset(DATASET_REPO)
    val_ds = (
        ds["validation"].shuffle(seed=42).select(range(min(sample_size, len(ds["validation"]))))
    )
    logger.info("Evaluation samples: %d", len(val_ds))

    client = OpenAI(base_url=api_base, api_key=api_key)

    base_outputs, adapter_outputs, references = [], [], []
    base_citations, adapter_citations = 0, 0

    for i, example in enumerate(val_ds):
        question = example.get("input") or example.get("instruction", "")
        reference = example.get("output", "")
        if not question:
            continue

        logger.info("[%d/%d] Generating...", i + 1, len(val_ds))

        base_out = generate_response(client, base_model, question)
        adapter_out = generate_response(client, adapter_model, question)

        base_outputs.append(base_out)
        adapter_outputs.append(adapter_out)
        references.append(reference)

        if has_citation(base_out):
            base_citations += 1
        if has_citation(adapter_out):
            adapter_citations += 1

    n = len(references)
    base_citation_rate = base_citations / n if n else 0
    adapter_citation_rate = adapter_citations / n if n else 0

    logger.info("Computing BERTScore...")
    from bert_score import score as bert_score

    _, _, base_bert_f1 = bert_score(base_outputs, references, lang="ko", verbose=False)
    _, _, adapter_bert_f1 = bert_score(adapter_outputs, references, lang="ko", verbose=False)

    logger.info("Computing ROUGE-L...")
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    base_rouge, adapter_rouge = [], []
    for b, a, r in zip(base_outputs, adapter_outputs, references):
        base_rouge.append(scorer.score(r, b)["rougeL"].fmeasure)
        adapter_rouge.append(scorer.score(r, a)["rougeL"].fmeasure)

    results = {
        "n_samples": n,
        "base_model": base_model,
        "adapter_model": adapter_model,
        "citation_rate": {
            "base": round(base_citation_rate, 4),
            "adapter": round(adapter_citation_rate, 4),
            "delta": round(adapter_citation_rate - base_citation_rate, 4),
        },
        "bertscore_f1": {
            "base": round(float(base_bert_f1.mean()), 4),
            "adapter": round(float(adapter_bert_f1.mean()), 4),
        },
        "rouge_l": {
            "base": round(sum(base_rouge) / len(base_rouge), 4),
            "adapter": round(sum(adapter_rouge) / len(adapter_rouge), 4),
        },
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="GovOn Legal Adapter Evaluation")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--base-model", default="LGAI-EXAONE/EXAONE-4.0-32B-AWQ")
    parser.add_argument("--adapter-model", default="legal-adapter")
    parser.add_argument(
        "--api-base", default=os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
    )
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    results = evaluate(
        sample_size=args.sample_size,
        base_model=args.base_model,
        adapter_model=args.adapter_model,
        api_base=args.api_base,
    )

    print("\n=== Evaluation Results ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
