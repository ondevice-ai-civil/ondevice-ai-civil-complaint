import wandb
import os
import pandas as pd
from tabulate import tabulate

def analyze_project():
    api = wandb.Api()
    entity = "umyun3"
    project = "GovOn-retrain-v2"
    path = f"{entity}/{project}"

    print(f"--- 분석 대상 프로젝트: {path} ---")

    try:
        # 1. 실행(Run) 요약 정보 조회
        runs = api.runs(path, order="-created_at")
        run_data = []
        for run in runs[:10]:
            run_data.append({
                "Name": run.name,
                "State": run.state,
                "Train_Loss": run.summary_metrics.get("train/loss"),
                "Eval_Loss": run.summary_metrics.get("eval/loss"),
                "LR": run.config.get("learning_rate"),
                "Epochs": run.config.get("num_train_epochs"),
                "Batch_Size": run.config.get("per_device_train_batch_size")
            })

        df = pd.DataFrame(run_data)
        print("\n[최근 학습 실행 요약]")
        print(tabulate(df, headers="keys", tablefmt="grid"))

        # 2. 성능 이슈 진단 (Loss 추이 분석)
        if not df.empty and df['Train_Loss'].notnull().any():
            best_run = df.loc[df['Train_Loss'].idxmin()]
            print(f"\n최저 Loss 달성 실행: {best_run['Name']} (Loss: {best_run['Train_Loss']})")
        
    except Exception as e:
        print(f"W&B API 접근 중 에러 발생: {e}")

if __name__ == "__main__":
    analyze_project()
