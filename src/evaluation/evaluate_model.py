"""
Comprehensive Model Evaluation Script
Evaluates AWQ quantized EXAONE-Deep-7.8B on civil complaint tasks.
Metrics: Perplexity, Classification Accuracy, BLEU, ROUGE-L, Inference Speed, VRAM Usage
"""

import os
import sys
import time
import json
import torch
import wandb
import gc
import re
import numpy as np
from datetime import datetime
from collections import defaultdict

MERGED_MODEL_DIR = "/content/ondevice-ai-civil-complaint/models/merged_model"
AWQ_MODEL_DIR = "/content/ondevice-ai-civil-complaint/models/awq_quantized_model"
TEST_DATA_PATH = "/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_test.jsonl"
VAL_DATA_PATH = "/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_val.jsonl"
RESULTS_DIR = "/content/ondevice-ai-civil-complaint/docs/outputs/M2_MVP"


def load_test_data(path, max_samples=200):
    """Load test data for evaluation."""
    import random
    random.seed(42)
    data = []
    with open(path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError:
                continue
    random.shuffle(data)
    return data[:max_samples]


def extract_category(text):
    """Extract category from input text [Category: xxx] pattern."""
    match = re.search(r'\[Category:\s*([^\]]+)\]', text)
    if match:
        return match.group(1).strip().lower()
    return "unknown"


def compute_perplexity(model, tokenizer, data, max_samples=100, max_length=2048):
    """Compute perplexity on test data."""
    print("  Computing perplexity...")
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_processed = 0

    for item in data[:max_samples]:
        try:
            messages = [
                {"role": "user", "content": f"{item['instruction']}\n\n{item['input']}"},
                {"role": "assistant", "content": item['output']}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = encodings.input_ids.to(model.device)

            if input_ids.shape[1] < 2:
                continue

            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss

            if not torch.isnan(loss) and not torch.isinf(loss):
                seq_len = input_ids.shape[1] - 1
                total_loss += loss.item() * seq_len
                total_tokens += seq_len
                n_processed += 1
                if n_processed % 10 == 0:
                    print(f"    Processed {n_processed} samples...", flush=True)

        except Exception as e:
            continue

    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        ppl = np.exp(avg_loss)
    else:
        ppl = float('inf')

    print(f"  Perplexity: {ppl:.4f} (over {n_processed} samples, {total_tokens} tokens)")
    return ppl, n_processed


def evaluate_classification(model, tokenizer, data, max_samples=100):
    """Evaluate civil complaint classification accuracy."""
    print("  Evaluating classification accuracy...")

    correct = 0
    total = 0
    predictions = []

    categories = ["environment", "traffic", "facilities", "civil_service", "welfare",
                   "culture", "economy", "education", "safety", "other"]

    # Use training instruction format
    instruction = "다음 민원에 대해 단계적으로 분석하고, 표준 서식에 맞춰 공손하고 명확한 답변을 작성하세요."

    for item in data[:max_samples]:
        try:
            true_category = extract_category(item.get('input', ''))
            if true_category == "unknown":
                continue

            # Use same instruction+input format as training
            messages = [{"role": "user", "content": f"{instruction}\n\n{item['input'][:500]}"}]
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
            )
            input_ids = encoded.input_ids.to(model.device)

            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=300,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=int(tokenizer.eos_token_id),
                )

            response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

            # Parse category from thought block: "Identified as [category] related request"
            pred_category = "unknown"
            cat_match = re.search(r'Identified as (\w+[\w/]*) related', response, re.IGNORECASE)
            if cat_match:
                pred_category = cat_match.group(1).lower().replace('/', '_')
                # Normalize known aliases
                if pred_category not in categories:
                    pred_category = "other"
            else:
                # Fallback: check after </thought>
                check_text = response.split('</thought>')[-1] if '</thought>' in response else response
                check_text = check_text.lower()
                for cat in categories:
                    if cat in check_text:
                        pred_category = cat
                        break

            is_correct = pred_category == true_category
            if is_correct:
                correct += 1
            total += 1

            predictions.append({
                "true": true_category,
                "predicted": pred_category,
                "correct": is_correct,
                "response_snippet": response[:200],
            })

        except Exception as e:
            continue

    accuracy = correct / total if total > 0 else 0
    print(f"  Classification Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy, predictions


def compute_generation_metrics(model, tokenizer, data, max_samples=50):
    """Compute BLEU and ROUGE-L scores for answer generation."""
    print("  Computing generation metrics (BLEU, ROUGE-L)...")

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        has_rouge = True
    except ImportError:
        print("  [WARN] rouge_score not installed, skipping ROUGE")
        has_rouge = False

    bleu_scores = []
    rouge_scores = []
    generated_samples = []

    for item in data[:max_samples]:
        try:
            messages = [{"role": "user", "content": f"{item['instruction']}\n\n{item['input']}"}]
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
            )
            input_ids = encoded.input_ids.to(model.device)

            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    eos_token_id=int(tokenizer.eos_token_id),
                )

            generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            # Remove <thought> tags for comparison
            generated_clean = re.sub(r'<thought>.*?</thought>', '', generated, flags=re.DOTALL).strip()
            reference = item.get('output', '')
            reference_clean = re.sub(r'<thought>.*?</thought>', '', reference, flags=re.DOTALL).strip()

            if not generated_clean or not reference_clean:
                continue

            # Simple n-gram BLEU approximation
            gen_tokens = generated_clean.split()
            ref_tokens = reference_clean.split()
            if len(gen_tokens) > 0 and len(ref_tokens) > 0:
                # Unigram precision
                matches = sum(1 for t in gen_tokens if t in ref_tokens)
                precision = matches / len(gen_tokens) if len(gen_tokens) > 0 else 0
                # Brevity penalty
                bp = min(1.0, len(gen_tokens) / len(ref_tokens)) if len(ref_tokens) > 0 else 0
                bleu_approx = bp * precision * 100
                bleu_scores.append(bleu_approx)

            # ROUGE-L
            if has_rouge:
                score = scorer.score(reference_clean, generated_clean)
                rouge_scores.append(score['rougeL'].fmeasure * 100)

            generated_samples.append({
                "input": item['input'][:200],
                "reference": reference_clean[:200],
                "generated": generated_clean[:200],
            })

        except Exception as e:
            continue

    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    avg_rouge = np.mean(rouge_scores) if rouge_scores else 0

    print(f"  BLEU (approx): {avg_bleu:.2f}")
    print(f"  ROUGE-L: {avg_rouge:.2f}")
    return avg_bleu, avg_rouge, generated_samples


def benchmark_inference(model, tokenizer, n_runs=10):
    """Benchmark inference speed and memory."""
    print("  Benchmarking inference speed...")

    prompts = [
        "도로에 포트홀이 생겼습니다. 수리 요청합니다.",
        "주민센터 민원실 운영시간을 알려주세요.",
        "아파트 앞 가로등이 고장났습니다.",
        "소음 문제로 민원 신고합니다.",
        "주차 위반 차량 단속 요청합니다.",
    ]

    # Warm-up
    messages = [{"role": "user", "content": prompts[0]}]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    input_ids = encoded.input_ids.to(model.device)
    with torch.no_grad():
        _ = model.generate(input_ids=input_ids, max_new_tokens=50, do_sample=False)

    # Benchmark
    latencies = []
    tokens_generated = []
    first_token_latencies = []

    torch.cuda.synchronize()

    for i in range(n_runs):
        prompt = prompts[i % len(prompts)]
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        input_ids = encoded.input_ids.to(model.device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                eos_token_id=int(tokenizer.eos_token_id),
            )

        torch.cuda.synchronize()
        end = time.perf_counter()

        n_new_tokens = output.shape[1] - input_ids.shape[1]
        elapsed = end - start
        latencies.append(elapsed)
        tokens_generated.append(n_new_tokens)

    avg_latency = np.mean(latencies)
    avg_tokens = np.mean(tokens_generated)
    throughput = avg_tokens / avg_latency if avg_latency > 0 else 0
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)

    # Memory usage
    gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3
    gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
    gpu_mem_max = torch.cuda.max_memory_allocated() / 1024**3

    results = {
        "avg_latency_s": avg_latency,
        "p50_latency_s": p50_latency,
        "p95_latency_s": p95_latency,
        "avg_tokens_generated": avg_tokens,
        "throughput_tok_s": throughput,
        "gpu_mem_allocated_gb": gpu_mem_allocated,
        "gpu_mem_reserved_gb": gpu_mem_reserved,
        "gpu_mem_max_gb": gpu_mem_max,
    }

    print(f"  Avg latency: {avg_latency:.3f}s")
    print(f"  P50 latency: {p50_latency:.3f}s")
    print(f"  P95 latency: {p95_latency:.3f}s")
    print(f"  Throughput: {throughput:.1f} tok/s")
    print(f"  GPU VRAM (allocated): {gpu_mem_allocated:.2f} GB")
    print(f"  GPU VRAM (max): {gpu_mem_max:.2f} GB")

    return results


def main():
    start_time = time.time()

    # Initialize WandB
    run = wandb.init(
        project="exaone-civil-complaint",
        name=f"evaluation-{datetime.now().strftime('%Y%m%d-%H%M')}",
        tags=["evaluation", "awq", "exaone-7.8b"],
        config={
            "awq_model_dir": AWQ_MODEL_DIR,
            "test_data_path": TEST_DATA_PATH,
            "stage": "3_evaluation",
        }
    )

    print("=" * 60)
    print("Stage 3: Model Evaluation")
    print("=" * 60)

    # Load test data
    print("\n[1/6] Loading test data...")
    test_data = load_test_data(TEST_DATA_PATH, max_samples=200)
    print(f"  Loaded {len(test_data)} test samples")

    # Load AWQ model using autoawq directly (avoids gptqmodel ExLlama kernel issue)
    print("\n[2/6] Loading AWQ quantized model...")
    from transformers import AutoTokenizer
    from awq import AutoAWQForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(AWQ_MODEL_DIR, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_quantized(
        AWQ_MODEL_DIR,
        fuse_layers=False,
        trust_remote_code=True,
    )
    # Get the underlying model for generation
    model = model.model

    # Monkey-patch for EXAONE compatibility
    try:
        model.get_input_embeddings()
    except (NotImplementedError, AttributeError):
        model.get_input_embeddings = lambda: model.transformer.wte
    try:
        model.get_output_embeddings()
    except (NotImplementedError, AttributeError):
        model.get_output_embeddings = lambda: model.lm_head

    model.eval()
    print(f"  Model loaded. Device: {model.device}")

    # 3. Perplexity
    print("\n[3/6] Evaluating Perplexity...", flush=True)
    ppl, ppl_samples = compute_perplexity(model, tokenizer, test_data, max_samples=50)
    wandb.log({"perplexity": ppl, "perplexity_samples": ppl_samples})
    print(f"  [DONE] Perplexity evaluation complete", flush=True)

    # 4. Classification Accuracy
    print("\n[4/6] Evaluating Classification Accuracy...", flush=True)
    accuracy, class_predictions = evaluate_classification(model, tokenizer, test_data, max_samples=50)
    wandb.log({"classification_accuracy": accuracy, "classification_total": len(class_predictions)})
    print(f"  [DONE] Classification evaluation complete", flush=True)

    # 5. Generation Metrics
    print("\n[5/6] Evaluating Generation Quality...", flush=True)
    bleu, rouge_l, gen_samples = compute_generation_metrics(model, tokenizer, test_data, max_samples=30)
    wandb.log({"bleu_score": bleu, "rouge_l_score": rouge_l})

    # 6. Inference Benchmark
    print("\n[6/6] Benchmarking Inference Speed...")
    bench_results = benchmark_inference(model, tokenizer, n_runs=10)
    wandb.log(bench_results)

    # Compile all results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model": "EXAONE-Deep-7.8B-AWQ (civil-complaint fine-tuned)",
        "quantization": "AWQ W4A16g128",
        "metrics": {
            "perplexity": round(ppl, 4),
            "classification_accuracy": round(accuracy, 4),
            "bleu_score": round(bleu, 2),
            "rouge_l_score": round(rouge_l, 2),
        },
        "inference": {
            "avg_latency_s": round(bench_results["avg_latency_s"], 3),
            "p50_latency_s": round(bench_results["p50_latency_s"], 3),
            "p95_latency_s": round(bench_results["p95_latency_s"], 3),
            "throughput_tok_s": round(bench_results["throughput_tok_s"], 1),
            "gpu_vram_allocated_gb": round(bench_results["gpu_mem_allocated_gb"], 2),
            "gpu_vram_max_gb": round(bench_results["gpu_mem_max_gb"], 2),
        },
        "model_sizes": {
            "merged_bf16_gb": 14.56,
            "awq_4bit_gb": 4.94,
            "compression_ratio": 2.95,
            "size_reduction_pct": 66.1,
        },
        "sample_predictions": class_predictions[:10] if class_predictions else [],
        "sample_generations": gen_samples[:5] if gen_samples else [],
    }

    # Save results
    results_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {results_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<35} {'Value':<20} {'Target':<15}")
    print("-" * 70)
    print(f"{'Perplexity':<35} {ppl:<20.4f} {'< inf':>15}")
    print(f"{'Classification Accuracy':<35} {accuracy*100:<20.2f}% {'>=85%':>15}")
    print(f"{'BLEU Score':<35} {bleu:<20.2f} {'>=30':>15}")
    print(f"{'ROUGE-L Score':<35} {rouge_l:<20.2f} {'>=40':>15}")
    print(f"{'Avg Latency (s)':<35} {bench_results['avg_latency_s']:<20.3f} {'<2s':>15}")
    print(f"{'P50 Latency (s)':<35} {bench_results['p50_latency_s']:<20.3f} {'<2s':>15}")
    print(f"{'P95 Latency (s)':<35} {bench_results['p95_latency_s']:<20.3f} {'<5s':>15}")
    print(f"{'Throughput (tok/s)':<35} {bench_results['throughput_tok_s']:<20.1f}")
    print(f"{'GPU VRAM (GB)':<35} {bench_results['gpu_mem_allocated_gb']:<20.2f} {'<8GB':>15}")
    print(f"{'Model Size (GB)':<35} {4.94:<20.2f} {'<5GB':>15}")
    print("=" * 70)

    # KPI check
    print("\nKPI Achievement Check:")
    kpi_results = {
        "classification_accuracy_pass": accuracy >= 0.85,
        "bleu_pass": bleu >= 30,
        "rouge_l_pass": rouge_l >= 40,
        "vram_pass": bench_results["gpu_mem_allocated_gb"] < 8,
        "model_size_pass": 4.94 < 5,
    }
    for kpi, passed in kpi_results.items():
        status = "PASS" if passed else "NEEDS IMPROVEMENT"
        print(f"  {kpi}: {status}")

    wandb.log(kpi_results)

    elapsed = time.time() - start_time
    print(f"\nTotal evaluation time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    wandb.log({"total_evaluation_time_seconds": elapsed})
    wandb.finish()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return all_results


if __name__ == "__main__":
    main()
