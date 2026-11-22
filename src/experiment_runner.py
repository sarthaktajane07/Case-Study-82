import os
import json
import csv
import time
from typing import Dict, Any, List
from api_client import NimbusApiClient

# System prompt for Nimbus Ridge Weather Bulletin Service
SYSTEM_PROMPT = (
    "You are the official weather bulletin forecaster for the Nimbus Ridge Weather Bulletin Service. "
    "Your job is to generate a concise, plain-language weather bulletin for hill-district villagers based strictly "
    "on the structured readings provided by forecasters. Include the temperature, rain probability, wind conditions, "
    "and any active advisory clearly. Do not invent any facts, specific road closures, power outage claims, or unmentioned "
    "hazards that are not directly supported by the input readings."
)

# Test inputs defined in Case Study 82
TEST_INPUTS = {
    "R1": "Max 24 C, rain 10%, wind 8 km/h, advisory: none",
    "R2": "Max 17 C, rain 85%, wind 35 km/h, advisory: landslide watch",
    "R3": "Max 29 C, rain 0%, wind 5 km/h, advisory: heat",
    "R4": "Max 12 C, rain 60%, wind 20 km/h, advisory: fog",
    "R5": "Max 21 C, rain 30%, wind 12 km/h, advisory: none",
}

# Parameter variations
TEMPERATURE_VALUES = [0.0, 0.3, 0.7, 1.0, 1.4]
TOP_P_VALUES = [0.3, 0.6, 0.9, 1.0]
MAX_TOKENS_VALUES = [40, 80, 160]
STOP_SEQUENCES = [
    ("none", None),
    ("double_newline", ["\n\n"]),
    ("advisory_colon", ["Advisory:"]),
]

DEFAULT_TEMP = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 160
NUM_REPETITIONS = 5

def run_all_experiments(
    model: str = "gemma4:e2b",
    output_dir: str = "/Users/sarthak/Downloads/Case_Study_82/data",
):
    os.makedirs(output_dir, exist_ok=True)
    raw_jsonl_path = os.path.join(output_dir, "experiments_raw.jsonl")
    raw_csv_path = os.path.join(output_dir, "experiments_raw.csv")

    client = NimbusApiClient(default_model=model)
    records = []

    print(f"=== Starting Nimbus Ridge Experiment Suite using Model: {model} ===")

    def log_and_save(experiment_type: str, param_varied: str, param_value: Any, input_id: str, rep: int, seed: Any, res: Dict[str, Any]):
        record = {
            "experiment_type": experiment_type,
            "param_varied": param_varied,
            "param_value": str(param_value),
            "input_id": input_id,
            "input_text": TEST_INPUTS[input_id],
            "repetition": rep,
            "seed": seed,
            "temperature": res["request_parameters"]["temperature"],
            "top_p": res["request_parameters"]["top_p"],
            "max_tokens": res["request_parameters"]["max_tokens"],
            "stop": str(res["request_parameters"]["stop"]),
            "finish_reason": res["finish_reason"],
            "prompt_tokens": res["token_usage"]["prompt_tokens"],
            "completion_tokens": res["token_usage"]["completion_tokens"],
            "total_tokens": res["token_usage"]["total_tokens"],
            "latency_ms": res["latency_ms"],
            "response_text": res["response_text"],
        }
        records.append(record)
        with open(raw_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 1. Temperature variations
    print("\n[1/5] Running Temperature Variations...")
    for temp in TEMPERATURE_VALUES:
        for input_id, input_text in TEST_INPUTS.items():
            for rep in range(1, NUM_REPETITIONS + 1):
                res = client.generate_bulletin(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=f"Input readings: {input_text}",
                    model=model,
                    temperature=temp,
                    top_p=DEFAULT_TOP_P,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    seed=None,
                )
                log_and_save("temperature_sweep", "temperature", temp, input_id, rep, None, res)
                print(f"  Temp={temp} | {input_id} | Rep {rep}/{NUM_REPETITIONS} -> tokens: {res['token_usage']['completion_tokens']}, finish: {res['finish_reason']}")

    # 2. Top-p variations
    print("\n[2/5] Running Top-P Variations...")
    for top_p in TOP_P_VALUES:
        for input_id, input_text in TEST_INPUTS.items():
            for rep in range(1, NUM_REPETITIONS + 1):
                res = client.generate_bulletin(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=f"Input readings: {input_text}",
                    model=model,
                    temperature=DEFAULT_TEMP,
                    top_p=top_p,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    seed=None,
                )
                log_and_save("top_p_sweep", "top_p", top_p, input_id, rep, None, res)
                print(f"  Top_p={top_p} | {input_id} | Rep {rep}/{NUM_REPETITIONS} -> tokens: {res['token_usage']['completion_tokens']}, finish: {res['finish_reason']}")

    # 3. Max tokens variations
    print("\n[3/5] Running Max Tokens Variations...")
    for max_tok in MAX_TOKENS_VALUES:
        for input_id, input_text in TEST_INPUTS.items():
            for rep in range(1, NUM_REPETITIONS + 1):
                res = client.generate_bulletin(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=f"Input readings: {input_text}",
                    model=model,
                    temperature=DEFAULT_TEMP,
                    top_p=DEFAULT_TOP_P,
                    max_tokens=max_tok,
                    seed=None,
                )
                log_and_save("max_tokens_sweep", "max_tokens", max_tok, input_id, rep, None, res)
                print(f"  MaxTokens={max_tok} | {input_id} | Rep {rep}/{NUM_REPETITIONS} -> tokens: {res['token_usage']['completion_tokens']}, finish: {res['finish_reason']}")

    # 4. Stop sequence variations
    print("\n[4/5] Running Stop Sequence Variations...")
    for stop_label, stop_seq in STOP_SEQUENCES:
        for input_id, input_text in TEST_INPUTS.items():
            for rep in range(1, NUM_REPETITIONS + 1):
                res = client.generate_bulletin(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=f"Input readings: {input_text}",
                    model=model,
                    temperature=DEFAULT_TEMP,
                    top_p=DEFAULT_TOP_P,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    stop=stop_seq,
                    seed=None,
                )
                log_and_save("stop_sequence_sweep", "stop_sequence", stop_label, input_id, rep, None, res)
                print(f"  Stop={stop_label} | {input_id} | Rep {rep}/{NUM_REPETITIONS} -> tokens: {res['token_usage']['completion_tokens']}, finish: {res['finish_reason']}")

    # 5. Seed / Repeat stability variations
    print("\n[5/5] Running Seed Stability Variations (Seed=42 vs Unseeded)...")
    for seed_val in [42, None]:
        seed_label = "fixed_42" if seed_val is not None else "unseeded"
        for input_id, input_text in TEST_INPUTS.items():
            for rep in range(1, NUM_REPETITIONS + 1):
                res = client.generate_bulletin(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=f"Input readings: {input_text}",
                    model=model,
                    temperature=DEFAULT_TEMP,
                    top_p=DEFAULT_TOP_P,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    seed=seed_val,
                )
                log_and_save("seed_stability_sweep", "seed", seed_label, input_id, rep, seed_val, res)
                print(f"  Seed={seed_label} | {input_id} | Rep {rep}/{NUM_REPETITIONS} -> tokens: {res['token_usage']['completion_tokens']}")

    # Write summary CSV
    if records:
        keys = list(records[0].keys())
        with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)

    print(f"\nAll experiments complete. Total runs logged: {len(records)}")
    print(f"JSONL saved to: {raw_jsonl_path}")
    print(f"CSV saved to: {raw_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma4:e2b", help="Model name")
    args = parser.parse_args()
    run_all_experiments(model=args.model)
