"""페르소나 기반 답변 품질 자동 평가기.

AgentManager의 에이전트 페르소나가 생성한 답변의 품질을
BERTScore와 ROUGE-L로 자동 측정한다.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import bert_score
import numpy as np
from loguru import logger
from rouge_score import rouge_scorer


class PersonaEvaluator:
    """BERTScore + ROUGE-L 기반 답변 품질 평가기.

    Attributes:
        threshold: BERTScore F1 합격 기준 임계값
        lang: BERTScore 평가 언어
    """

    def __init__(self, threshold: float = 0.70, lang: str = "ko") -> None:
        """PersonaEvaluator 초기화.

        Args:
            threshold: BERTScore F1 합격 기준 (기본값 0.70)
            lang: BERTScore 평가 대상 언어 (기본값 "ko")
        """
        self.threshold = threshold
        self.lang = lang
        self._rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    def evaluate_single(self, generated: str, reference: str) -> dict:
        """단일 답변 품질 평가.

        Args:
            generated: 모델이 생성한 답변 텍스트
            reference: 정답(참조) 텍스트

        Returns:
            평가 결과 딕셔너리:
                - bert_score_f1: BERTScore F1 점수
                - rouge_l: ROUGE-L F-measure
                - passed: 합격 여부 (threshold 기준)
        """
        # BERTScore 계산
        P, R, F1 = bert_score.score([generated], [reference], lang=self.lang)
        bert_f1 = float(F1[0].item())

        # ROUGE-L 계산
        rouge_result = self._rouge_scorer.score(reference, generated)
        rouge_l = float(rouge_result["rougeL"].fmeasure)

        passed = bert_f1 >= self.threshold

        return {
            "bert_score_f1": bert_f1,
            "rouge_l": rouge_l,
            "passed": passed,
        }

    def evaluate_batch(self, generations: list[str], references: list[str]) -> dict:
        """배치 답변 품질 평가.

        Args:
            generations: 모델이 생성한 답변 텍스트 리스트
            references: 정답(참조) 텍스트 리스트

        Returns:
            배치 평가 결과 딕셔너리:
                - bert_score_f1: 평균/중앙값/표준편차
                - rouge_l: 평균/중앙값/표준편차
                - acceptance_rate: 채택률 (threshold 이상인 비율)
                - num_samples: 평가 샘플 수
                - num_passed: 합격 샘플 수

        Raises:
            ValueError: generations와 references의 길이가 다를 경우
        """
        if len(generations) != len(references):
            raise ValueError(
                f"generations({len(generations)})와 "
                f"references({len(references)})의 길이가 다릅니다."
            )

        if not generations:
            return {
                "bert_score_f1": {"mean": 0.0, "median": 0.0, "std": 0.0},
                "rouge_l": {"mean": 0.0, "median": 0.0, "std": 0.0},
                "acceptance_rate": 0.0,
                "num_samples": 0,
                "num_passed": 0,
            }

        # BERTScore 배치 계산
        logger.info(f"BERTScore 배치 계산 중 ({len(generations)}건)...")
        P, R, F1 = bert_score.score(generations, references, lang=self.lang)
        bert_f1_scores = [float(f) for f in F1.tolist()]

        # ROUGE-L 배치 계산
        logger.info(f"ROUGE-L 배치 계산 중 ({len(generations)}건)...")
        rouge_l_scores = []
        for gen, ref in zip(generations, references):
            result = self._rouge_scorer.score(ref, gen)
            rouge_l_scores.append(float(result["rougeL"].fmeasure))

        bert_arr = np.array(bert_f1_scores)
        rouge_arr = np.array(rouge_l_scores)
        num_passed = int(np.sum(bert_arr >= self.threshold))

        return {
            "bert_score_f1": {
                "mean": float(np.mean(bert_arr)),
                "median": float(np.median(bert_arr)),
                "std": float(np.std(bert_arr)),
            },
            "rouge_l": {
                "mean": float(np.mean(rouge_arr)),
                "median": float(np.median(rouge_arr)),
                "std": float(np.std(rouge_arr)),
            },
            "acceptance_rate": num_passed / len(generations),
            "num_samples": len(generations),
            "num_passed": num_passed,
        }

    def generate_report(self, results: dict, output_path: str) -> None:
        """Markdown 형식 평가 리포트 생성.

        Args:
            results: evaluate_batch()의 반환값
            output_path: 리포트 저장 경로
        """
        bert = results.get("bert_score_f1", {})
        rouge = results.get("rouge_l", {})

        lines = [
            "# 페르소나 답변 품질 평가 리포트",
            "",
            f"**평가 일시**: {datetime.now().isoformat()}",
            f"**평가 샘플 수**: {results.get('num_samples', 0)}",
            f"**합격 기준**: BERTScore F1 >= {self.threshold}",
            "",
            "## 평가 결과",
            "",
            "| 지표 | 평균 | 중앙값 | 표준편차 |",
            "| --- | --- | --- | --- |",
            f"| BERTScore F1 | {bert.get('mean', 0):.4f} | {bert.get('median', 0):.4f} | {bert.get('std', 0):.4f} |",
            f"| ROUGE-L | {rouge.get('mean', 0):.4f} | {rouge.get('median', 0):.4f} | {rouge.get('std', 0):.4f} |",
            "",
            "## 채택률",
            "",
            f"| 항목 | 값 |",
            f"| --- | --- |",
            f"| 합격 샘플 수 | {results.get('num_passed', 0)} / {results.get('num_samples', 0)} |",
            f"| 채택률 | {results.get('acceptance_rate', 0):.2%} |",
            "",
            f"리포트 생성: {datetime.now().isoformat()}",
        ]

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"평가 리포트 저장: {output_path}")
