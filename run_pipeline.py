import os
import sys
import time
import subprocess
import requests

def check_ollama_ready(model_name: str = "gemma4:e2b", max_wait_sec: int = 1200) -> bool:
    print(f"Checking if Ollama service is up and '{model_name}' is loaded...")
    start_time = time.time()
    while time.time() - start_time < max_wait_sec:
        try:
            res = requests.get("http://localhost:11434/api/tags", timeout=5)
            if res.status_code == 200:
                models = [m["name"] for m in res.json().get("models", [])]
                if any(model_name in m for m in models):
                    print(f"Found model '{model_name}' ready in Ollama!")
                    return True
                else:
                    print(f"Waiting for '{model_name}' to finish pulling... (Available: {models})")
            else:
                print("Ollama API responded with non-200 status code.")
        except Exception as e:
            print(f"Waiting for Ollama service to respond... ({e})")
        time.sleep(10)
    return False

def main():
    model = "gemma4:e2b"
    print("=====================================================================")
    print(" Nimbus Ridge Weather Bulletin Service - Automated Experiment Pipeline")
    print(f" Target Model: {model}")
    print("=====================================================================")

    if not check_ollama_ready(model):
        print(f"ERROR: Model '{model}' not ready in Ollama after wait timeout.")
        sys.exit(1)

    # 1. Run experiments
    print("\n>>> Step 1: Executing Controlled Experiment Grid...")
    t0 = time.time()
    subprocess.run([sys.executable, "src/experiment_runner.py", "--model", model], check=True)
    print(f"Experiment execution completed in {time.time() - t0:.1f}s")

    # 2. Evaluate metrics
    print("\n>>> Step 2: Evaluating Diversity, Faithfulness, and Length Metrics...")
    subprocess.run([sys.executable, "src/evaluate_metrics.py"], check=True)

    # 3. Plot results
    print("\n>>> Step 3: Generating Visualizations and Charts...")
    subprocess.run([sys.executable, "src/plot_results.py"], check=True)

    print("\n=====================================================================")
    print(" ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print(" Artifacts generated:")
    print("  - Raw Logs: data/experiments_raw.jsonl, data/experiments_raw.csv")
    print("  - Summary Metrics: data/summary_metrics.csv")
    print("  - Faithfulness Audit: data/faithfulness_sample_20.csv")
    print("  - Figures: figures/fig1_diversity_vs_temperature_topp.png")
    print("             figures/fig2_faithfulness_vs_temperature.png")
    print("             figures/fig3_length_and_truncation_vs_maxtokens.png")
    print("             figures/fig4_stop_sequences_and_seed_stability.png")
    print("  - Report: REPORT.md")
    print("=====================================================================")

if __name__ == "__main__":
    main()
