"""Lean Startup 실험: RAG 활성화/비활성화 A/B 비교 평가.

v2_test.jsonl 전체 데이터에 대해 RAG on/off 두 조건에서 답변 품질을 비교하고
BERTScore F1 >= 0.70 기준으로 Pivot/Persevere 판단을 자동화한다.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import bert_score
import numpy as np
import torch
import wandb
import yaml
from loguru import logger
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.evaluation.persona_evaluator import PersonaEvaluator

# 2024-12-11: transformers v5 업데이트(2026-02-06) 이전 마지막 호환 revision
EXAONE_REVISION = "0ff6b5ec7c13b049b253a16a889aa269e6b79a94"

BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
DEFAULT_ADAPTER_ID = "umyunsang/civil-complaint-exaone-lora"
DEFAULT_TEST_DATA_PATH = "data/processed/v2_test.jsonl"
DEFAULT_CONFIG_PATH = "configs/experiment_config.yaml"


def parse_args() -> argparse.Namespace:
    """CLI 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="Lean Startup RAG A/B 비교 실험",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.getenv("MODEL_PATH", DEFAULT_ADAPTER_ID),
        help="모델(LoRA 어댑터) 경로 또는 ID",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.getenv("DATA_PATH", DEFAULT_TEST_DATA_PATH),
        help="테스트 데이터 JSONL 경로",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiment_results",
        help="결과 출력 디렉토리",
    )
    parser.add_argument(
        "--gpu_utilization",
        type=float,
        default=float(os.getenv("GPU_UTILIZATION", "0.7")),
        help="GPU 활용률",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="최대 샘플 수 (None=전체)",
    )
    parser.add_argument(
        "--bert_threshold",
        type=float,
        default=0.70,
        help="BERTScore Pivot/Persevere 임계값",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="실험 설정 YAML 파일 경로",
    )
    return parser.parse_args()


def load_test_data(data_path: str, max_samples: int | None = None) -> list[dict]:
    """v2_test.jsonl 로드."""
    if not os.path.exists(data_path):
        logger.error(f"테스트 데이터를 찾을 수 없습니다: {data_path}")
        raise FileNotFoundError(f"테스트 데이터 경로 없음: {data_path}")

    test_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))

    if max_samples is not None:
        test_data = test_data[:max_samples]

    logger.info(f"테스트 데이터 {len(test_data)}건 로드 완료")
    return test_data


def load_model(model_path: str):
    """EXAONE 모델 + LoRA 어댑터 로드."""
    from peft import PeftModel

    logger.info("모델 로딩 시작...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        revision=EXAONE_REVISION,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        revision=EXAONE_REVISION,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    logger.info("모델 로딩 완료")
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    input_text: str,
    use_rag: bool = False,
    max_new_tokens: int = 256,
) -> tuple[str, float]:
    """단일 입력에 대해 답변 생성 및 지연시간 측정.

    Args:
        model: 로드된 모델
        tokenizer: 토크나이저
        input_text: 사용자 입력 텍스트
        use_rag: RAG 파이프라인 사용 여부
        max_new_tokens: 최대 생성 토큰 수

    Returns:
        (생성된 답변, 지연시간(초)) 튜플
    """
    if use_rag:
        system_prompt = (
            "당신은 민원 상담 전문가입니다. "
            "제공된 참고 자료를 활용하여 정확하고 친절하게 답변하세요."
        )
    else:
        system_prompt = (
            "당신은 민원 상담 전문가입니다. " "학습된 지식을 활용하여 정확하고 친절하게 답변하세요."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    latency = time.perf_counter() - start

    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )
    return generated.strip(), latency


def run_condition(
    model,
    tokenizer,
    test_data: list[dict],
    use_rag: bool,
    condition_name: str,
) -> dict:
    """단일 실험 조건 실행.

    Args:
        model: 로드된 모델
        tokenizer: 토크나이저
        test_data: 테스트 데이터 리스트
        use_rag: RAG 파이프라인 사용 여부
        condition_name: 조건 이름 (로깅용)

    Returns:
        실험 결과 딕셔너리
    """
    logger.info(f"[{condition_name}] 실험 시작 (샘플 {len(test_data)}건)")

    os.environ["USE_RAG_PIPELINE"] = "true" if use_rag else "false"

    generations = []
    references = []
    latencies = []

    for idx, item in enumerate(test_data):
        input_text = item.get("input") or item.get("text", "")[:500]
        ref_text = item.get("output") or item.get("answer", "")

        generated, latency = generate_answer(model, tokenizer, input_text, use_rag=use_rag)

        generations.append(generated)
        references.append(ref_text.strip())
        latencies.append(latency)

        if (idx + 1) % 100 == 0:
            logger.info(f"[{condition_name}] {idx + 1}/{len(test_data)} 처리 완료")

    # BERTScore 계산
    logger.info(f"[{condition_name}] BERTScore 계산 중...")
    P, R, F1 = bert_score.score(generations, references, lang="ko")
    bert_f1_scores = F1.tolist()

    # ROUGE-L 계산
    logger.info(f"[{condition_name}] ROUGE-L 계산 중...")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l_scores = []
    for gen, ref in zip(generations, references):
        score = scorer.score(ref, gen)
        rouge_l_scores.append(score["rougeL"].fmeasure)

    # 통계 계산
    evaluator = PersonaEvaluator()
    batch_result = evaluator.evaluate_batch(generations, references)

    avg_bert_f1 = float(np.mean(bert_f1_scores))
    avg_rouge_l = float(np.mean(rouge_l_scores))
    avg_latency = float(np.mean(latencies))

    result = {
        "condition": condition_name,
        "use_rag": use_rag,
        "num_samples": len(test_data),
        "metrics": {
            "bert_score_f1": {
                "mean": avg_bert_f1,
                "median": float(np.median(bert_f1_scores)),
                "std": float(np.std(bert_f1_scores)),
                "min": float(np.min(bert_f1_scores)),
                "max": float(np.max(bert_f1_scores)),
            },
            "rouge_l": {
                "mean": avg_rouge_l,
                "median": float(np.median(rouge_l_scores)),
                "std": float(np.std(rouge_l_scores)),
                "min": float(np.min(rouge_l_scores)),
                "max": float(np.max(rouge_l_scores)),
            },
            "latency": {
                "mean": avg_latency,
                "median": float(np.median(latencies)),
                "std": float(np.std(latencies)),
            },
        },
        "acceptance_rate": batch_result["acceptance_rate"],
        "bert_f1_scores": bert_f1_scores,
        "rouge_l_scores": rouge_l_scores,
    }

    logger.info(
        f"[{condition_name}] 완료 - "
        f"BERTScore F1: {avg_bert_f1:.4f}, "
        f"ROUGE-L: {avg_rouge_l:.4f}, "
        f"채택률: {batch_result['acceptance_rate']:.2%}"
    )

    return result


def make_decision(rag_on_result: dict, threshold: float) -> dict:
    """Pivot/Persevere 판단.

    Args:
        rag_on_result: RAG 활성화 조건의 실험 결과
        threshold: BERTScore F1 임계값

    Returns:
        판단 결과 딕셔너리
    """
    avg_bert_f1 = rag_on_result["metrics"]["bert_score_f1"]["mean"]

    if avg_bert_f1 >= threshold:
        decision = "Persevere"
        action = "현재 RAG 전략 유지, 추가 최적화 진행"
        reason = (
            f"RAG 활성화 조건의 평균 BERTScore F1({avg_bert_f1:.4f})이 "
            f"임계값({threshold})을 충족합니다."
        )
    else:
        decision = "Pivot"
        action = "RAG 검색 품질 개선 또는 프롬프트 엔지니어링 재설계"
        reason = (
            f"RAG 활성화 조건의 평균 BERTScore F1({avg_bert_f1:.4f})이 "
            f"임계값({threshold}) 미만입니다."
        )

    return {
        "decision": decision,
        "action": action,
        "reason": reason,
        "threshold": threshold,
        "actual_score": avg_bert_f1,
    }


def save_json_report(results: dict, output_path: str) -> None:
    """JSON 형식 실험 결과 저장."""
    # bert_f1_scores, rouge_l_scores는 용량이 크므로 별도 저장
    report = {
        "experiment": results["experiment"],
        "conditions": [],
        "decision": results["decision"],
        "timestamp": results["timestamp"],
    }
    for cond in results["conditions"]:
        cond_copy = {k: v for k, v in cond.items() if k not in ("bert_f1_scores", "rouge_l_scores")}
        report["conditions"].append(cond_copy)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON 리포트 저장: {output_path}")


def save_markdown_report(results: dict, output_path: str) -> None:
    """Markdown 형식 실험 리포트 생성."""
    decision = results["decision"]
    conditions = results["conditions"]

    lines = [
        "# Lean Startup RAG A/B 실험 결과",
        "",
        f"**실험 일시**: {results['timestamp']}",
        f"**테스트 데이터**: {results['experiment']['data_path']}",
        "",
        "## 실험 조건별 결과",
        "",
        "| 지표 | RAG 활성화 | RAG 비활성화 |",
        "| --- | --- | --- |",
    ]

    rag_on = next(c for c in conditions if c["use_rag"])
    rag_off = next(c for c in conditions if not c["use_rag"])

    metrics_rows = [
        (
            "BERTScore F1 (평균)",
            f"{rag_on['metrics']['bert_score_f1']['mean']:.4f}",
            f"{rag_off['metrics']['bert_score_f1']['mean']:.4f}",
        ),
        (
            "BERTScore F1 (중앙값)",
            f"{rag_on['metrics']['bert_score_f1']['median']:.4f}",
            f"{rag_off['metrics']['bert_score_f1']['median']:.4f}",
        ),
        (
            "BERTScore F1 (표준편차)",
            f"{rag_on['metrics']['bert_score_f1']['std']:.4f}",
            f"{rag_off['metrics']['bert_score_f1']['std']:.4f}",
        ),
        (
            "ROUGE-L (평균)",
            f"{rag_on['metrics']['rouge_l']['mean']:.4f}",
            f"{rag_off['metrics']['rouge_l']['mean']:.4f}",
        ),
        (
            "ROUGE-L (중앙값)",
            f"{rag_on['metrics']['rouge_l']['median']:.4f}",
            f"{rag_off['metrics']['rouge_l']['median']:.4f}",
        ),
        (
            "답변 채택률",
            f"{rag_on['acceptance_rate']:.2%}",
            f"{rag_off['acceptance_rate']:.2%}",
        ),
        (
            "평균 지연시간",
            f"{rag_on['metrics']['latency']['mean']:.2f}s",
            f"{rag_off['metrics']['latency']['mean']:.2f}s",
        ),
    ]

    for name, on_val, off_val in metrics_rows:
        lines.append(f"| {name} | {on_val} | {off_val} |")

    # 판단 결과
    decision_emoji = "O" if decision["decision"] == "Persevere" else "X"
    lines.extend(
        [
            "",
            "## Pivot/Persevere 판단",
            "",
            f"| 항목 | 값 |",
            f"| --- | --- |",
            f"| 판단 | **{decision['decision']}** ({decision_emoji}) |",
            f"| 기준 | BERTScore F1 >= {decision['threshold']} |",
            f"| 실측값 | {decision['actual_score']:.4f} |",
            f"| 권장 조치 | {decision['action']} |",
            "",
            f"> {decision['reason']}",
            "",
            f"실험 완료: {results['timestamp']}",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Markdown 리포트 저장: {output_path}")


def main() -> None:
    """메인 실험 실행 함수."""
    args = parse_args()

    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 실험 설정 로드
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"실험 설정 로드: {args.config}")

    # wandb 초기화
    wandb_mode = "online" if os.getenv("WANDB_API_KEY") else "offline"
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "GovOn"),
        name=f"lean-experiment-{datetime.now().strftime('%m%d-%H%M')}",
        mode=wandb_mode,
        config={
            "experiment": "lean-startup-rag-comparison",
            "bert_threshold": args.bert_threshold,
            "max_samples": args.max_samples,
            "gpu_utilization": args.gpu_utilization,
        },
    )

    # 데이터 로드
    test_data = load_test_data(args.data_path, args.max_samples)

    # 모델 로드
    model, tokenizer = load_model(args.model_path)

    # Condition A: RAG 활성화
    rag_on_result = run_condition(
        model, tokenizer, test_data, use_rag=True, condition_name="rag_enabled"
    )
    wandb.log(
        {
            "rag_on/bert_score_f1": rag_on_result["metrics"]["bert_score_f1"]["mean"],
            "rag_on/rouge_l": rag_on_result["metrics"]["rouge_l"]["mean"],
            "rag_on/acceptance_rate": rag_on_result["acceptance_rate"],
            "rag_on/avg_latency": rag_on_result["metrics"]["latency"]["mean"],
        }
    )

    # Condition B: RAG 비활성화
    rag_off_result = run_condition(
        model, tokenizer, test_data, use_rag=False, condition_name="rag_disabled"
    )
    wandb.log(
        {
            "rag_off/bert_score_f1": rag_off_result["metrics"]["bert_score_f1"]["mean"],
            "rag_off/rouge_l": rag_off_result["metrics"]["rouge_l"]["mean"],
            "rag_off/acceptance_rate": rag_off_result["acceptance_rate"],
            "rag_off/avg_latency": rag_off_result["metrics"]["latency"]["mean"],
        }
    )

    # Pivot/Persevere 판단
    decision = make_decision(rag_on_result, args.bert_threshold)
    wandb.log(
        {
            "decision": 1 if decision["decision"] == "Persevere" else 0,
            "decision_score": decision["actual_score"],
            "decision_threshold": decision["threshold"],
        }
    )

    # 최종 결과 조합
    experiment_results = {
        "experiment": {
            "name": config.get("experiment", {}).get("name", "lean-startup-rag-comparison"),
            "version": config.get("experiment", {}).get("version", "1.0"),
            "model_path": args.model_path,
            "data_path": args.data_path,
            "num_samples": len(test_data),
            "bert_threshold": args.bert_threshold,
        },
        "conditions": [rag_on_result, rag_off_result],
        "decision": decision,
        "timestamp": datetime.now().isoformat(),
    }

    # 리포트 저장
    json_path = str(output_dir / "experiment_results.json")
    md_path = str(output_dir / "experiment_report.md")

    save_json_report(experiment_results, json_path)
    save_markdown_report(experiment_results, md_path)

    # 최종 결과 로깅
    logger.info(f"실험 완료 - 판단: {decision['decision']}")
    logger.info(f"  RAG on  BERTScore F1: {rag_on_result['metrics']['bert_score_f1']['mean']:.4f}")
    logger.info(f"  RAG off BERTScore F1: {rag_off_result['metrics']['bert_score_f1']['mean']:.4f}")
    logger.info(f"  임계값: {args.bert_threshold}")

    wandb.finish()


if __name__ == "__main__":
    main()
