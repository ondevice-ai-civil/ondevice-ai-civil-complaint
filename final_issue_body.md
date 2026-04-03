# [TASK] EXAONE-Deep-7.8B 파인튜닝 고도화 및 민원-공문서 통합 학습 체계 구축

## 1. 개요
기존 v2 파인튜닝 과정에서 발생한 기술적 한계를 극복하고, 행정안전부 '정부 공문서 AI 학습데이터'와 기존 '지자체 민원 처리 데이터'를 통합하여 모델의 공공 업무 수행 능력을 극대화함. GPU 자원 효율성과 배포 편의성을 위해 GitHub Actions 기반의 자동화 학습(CI) 및 HF Hub 배포(CD) 파이프라인을 구축함.

## 2. v2 학습 과정의 문제점 및 해결 방안 (W&B 분석 결과)

### 🚀 기술적 도전 과제 (Problems)
- **VRAM 부족 (OOM)**: 7.8B 모델 학습 시 L4(24GB) 환경에서 평가(Evaluation) 시 VRAM 임계치 도달.
- **라이브러리 호환성**: `transformers 5.3.0` 버전에서 `check_model_inputs` 삭제로 인한 EXAONE 모델 로딩 에러.
- **모델 구조적 결함**: PEFT 연동 시 `get_input_embeddings` 메서드 부재로 인한 `NotImplementedError` 발생.
- **평가 지표의 부재**: 단순 Loss 감소 외에 실제 공문서 문체(개조식)에 대한 정량적 검증 부족.

### ✅ 해결 전략 (Solutions)
- **런타임 업그레이드**: Google Colab A100(40GB) 또는 GitHub GPU Runner 환경으로 전환하여 `per_device_eval_batch_size=4` 확보.
- **Compatibility Patches**:
    - `transformers` 라이브러리 내 `generic.py` 수동 패치 적용 (`check_model_inputs` 더미 함수 추가).
    - `train_qlora.py` 내 `get_input_embeddings` 및 `get_output_embeddings` 동적 몽키 패치 적용.
- **메모리 최적화**: `paged_adamw_8bit` 옵티마이저 및 `bf16/tf32` 혼합 정밀도 학습 적용.
- **평가 표준화**: `SacreBLEU`, `ROUGE-L`, `BERTScore`를 도입하여 공문서 특유의 문체 변환 성능 검증.

## 3. 통합 데이터 수집 및 학습 전략 (Civil + Public Doc)

### 📊 데이터 수집 및 병합 파이프라인
- **행안부 공문서 (Public Doc)**: `getDocAll` API를 통해 ZIP 파일을 자동 다운로드하여 JSONL로 파싱 (보도자료, 연설문, 발간사 등 5종).
- **기존 민원 데이터 (Civil Complaint)**: 기존에 정제된 7만 건의 민원 질의응답 데이터를 로드.
- **병합 전략**: 두 데이터셋을 1:N 비율로 섞고 무작위 셔플링(Shuffle)하여 편향되지 않은 학습 데이터셋 구성.

### 🛠 데이터 전처리 로직 강화
- **Instruction Tuning**: 공문서 요약/재구성 태스크를 위해 "다음 공문서를 개조식으로 요약하세요" 등의 인스트럭션을 자동으로 생성하여 데이터셋 품질 향상.

## 4. GitHub Actions 기반 CI/CD 자동화 (GPU Runner)

- **Workflow**: `.github/workflows/train_exaone_gpu.yml`
- **주요 기능**:
    1. **Data**: 행안부 API 연동 및 민원 데이터 병합 자동화.
    2. **Train**: GPU 러너에서 QLoRA 학습 수행 (v2 최적화 하이퍼파라미터 적용).
    3. **Deploy**: 학습 완료 후 Hugging Face Hub에 모델 자동 업로드 (`GovOn/EXAONE-Deep-7.8B-Gov-v2`).

## 5. 핵심 변경 파일 (Core Files)
- `src/training/train_qlora.py`: HF 업로드 로직 및 EXAONE 호환성 패치 반영.
- `scripts/collect_data/collect_public_docs.py`: ZIP 기반 대량 수집 및 LLM 포맷 변환 로직 고도화.
- `.github/workflows/train_exaone_gpu.yml`: GPU 기반 통합 학습 및 배포 자동화.

## 6. 향후 일정
- [ ] GitHub Actions GPU 러너 활성화 및 통합 학습 테스트 수행
- [ ] Hugging Face Hub에 v2 모델 공개 및 추론 성능 벤치마크
- [ ] W&B 리포트를 통한 v2 대비 성능 비교 (Loss/Metrics)
