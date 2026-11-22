# Nimbus Ridge Weather Bulletin Service (Case Study no. 82)

## Course Outcome: CO1 | Topic: Calling an LLM API and Studying Sampling Parameters

This repository contains the complete experimental framework, source code, data pipelines, evaluation suite, and production recommendations for **Case Study no. 82: Nimbus Ridge Weather Bulletin Service**.

### System Architecture & Components
1. **API Client (`src/api_client.py`)**:
   - Programmatic chat-completions client interfacing with local LLM (`gemma4:e2b` via Ollama OpenAI-compatible endpoint `http://localhost:11434/v1`).
   - Implements exponential backoff retry logic, timeout handling, and structured telemetry logging (prompt, latency, token usage, finish reason).
2. **Experiment Runner (`src/experiment_runner.py`)**:
   - Executes controlled parameter sweeps over standard readings $R_1 - R_5$ with fixed system prompt.
   - Varied parameters:
     - Temperature: $0.0, 0.3, 0.7, 1.0, 1.4$ (5 runs each)
     - Nucleus sampling (top-p): $0.3, 0.6, 0.9, 1.0$ (5 runs each)
     - Maximum tokens: $40, 80, 160$ (5 runs each)
     - Stop sequences: None, `\n\n`, `Advisory:` (5 runs each)
     - Seed reproducibility: Fixed seed (`42`) vs stochastic unseeded (5 runs each)
   - Outputs persisted in `data/experiments_raw.jsonl` and `data/experiments_raw.csv`.
3. **Metric Evaluator (`src/evaluate_metrics.py`)**:
   - **Diversity**: Normalized pairwise Levenshtein edit distance and Distinct-$n$ gram ratios (Distinct-1, Distinct-2).
   - **Faithfulness**: Detection and counting of ungrounded/invented facts (e.g. road closures, power outages, exaggerated rain/wind) on a stratified sample of 20 outputs.
   - **Length & Truncation**: Mean/std completion tokens, word counts, and truncation frequency (`finish_reason == 'length'`).
4. **Visualization Suite (`src/plot_results.py`)**:
   - Publication-quality charts saved to `figures/`.
5. **Master Orchestrator (`run_pipeline.py`)**:
   - End-to-end execution script verifying model readiness, running experiments, evaluating metrics, and generating plots.

### Quick Start
```bash
# 1. Ensure Ollama is running and gemma4:e2b is downloaded
ollama run gemma4:e2b

# 2. Run the end-to-end pipeline
python3 run_pipeline.py
```
