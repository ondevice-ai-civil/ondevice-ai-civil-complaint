# EXAONE 모델 다운로드 로그

- **모델명**: [LGAI-EXAONE/EXAONE-Deep-7.8B](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-7.8B)
- **다운로드 일시**: 2026-03-05
- **저장 경로**: HuggingFace Cache (Local)
- **검증 내용**:
  - `safetensors` 가중치 로드 성공
  - `tokenizer` (32k context) 로드 성공
  - `trust_remote_code=True` 환경에서 모델 아키텍처(ExaoneForCausalLM) 정상 인식
- **특이 사항**: 
  - `transformers 5.3.0` dev 버전에서의 호환성 문제로 `check_model_inputs` 및 `get_input_embeddings` 몽키 패치 적용 후 로드함.
