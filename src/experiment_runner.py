import os
import json
import csv
import time
from typing import Dict, Any, List, Set, Tuple
from api_client import NimbusApiClient

SYSTEM_PROMPT = (
    "You are the official weather bulletin forecaster for the Nimbus Ridge Weather Bulletin Service. "
    "Your job is to generate a concise, plain-language weather bulletin for hill-district villagers based strictly "
    "on the structured readings provided by forecasters. Include the temperature, rain probability, wind conditions, "
    "and any active advisory clearly. Do not invent any facts, specific road closures, power outage claims, or unmentioned "
    "hazards that are not directly supported by the input readings."
)

TEST_INPUTS = {
    "R1": "Max 24 C, rain 10%, wind 8 km/h, advisory: none",
    "R2": "Max 17 C, rain 85%, wind 35 km/h, advisory: landslide watch",
    "R3": "Max 29 C, rain 0%, wind 5 km/h, advisory: heat",
    "R4": "Max 12 C, rain 60%, wind 20 km/h, advisory: fog",
    "R5": "Max 21 C, rain 30%, wind 12 km/h, advisory: none",
}

DEFAULT_TEMP = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 160
NUM_REPETITIONS = 5

def load_existing_signatures(jsonl_path: str) -> Tuple[List[Dict[str, Any]], Set[str]]:
    existing_records = []
    signatures = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    existing_records.append(rec)
                    sig = f"{rec['experiment_type']}_{rec['param_varied']}_{rec['param_value']}_{rec['input_id']}_{rec['repetition']}_{rec.get('seed')}"
                    signatures.add(sig)
    return existing_records, signatures

def run_all_experiments(
    model: str = "gemma2:2b",
    output_dir: str = "/Users/sarthak/Downloads/Case_Study_82/data",
):
    os.makedirs(output_dir, exist_ok=True)
    raw_jsonl_path = os.path.join(output_dir, "experiments_raw.jsonl")
    raw_csv_path = os.path.join(output_dir, "experiments_raw.csv")

    existing_records, existing_sigs = load_existing_signatures(raw_jsonl_path)
    records = list(existing_records)
    print(f"Loaded {len(records)} existing runs from log.")

    client = NimbusApiClient(default_model=model)

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

    # 1. Temperature sweep (R1 primary, plus R2-R5)
    print("\n[1/5] Temperature Sweep (0.0, 0.3, 0.7, 1.0, 1.4)...")
    for temp in [0.0, 0.3, 0.7, 1.0, 1.4]:
        for input_id in ["R1", "R2", "R3", "R4", "R5"]:
            for rep in range(1, NUM_REPETITIONS + 1):
                sig = f"temperature_sweep_temperature_{temp}_{input_id}_{rep}_None"
                if sig in existing_sigs:
                    continue
                res = client.generate_bulletin(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=f"Input readings: {TEST_INPUTS[input_id]}",
                    model=model,
                    temperature=temp,
                    top_p=DEFAULT_TOP_P,
                    max_tokens=DEFAULT_MAX_TOKENS,
                )
                log_and_save("temperature_sweep", "temperature", temp, input_id, rep, None, res)
                print(f"  Temp={temp} | {input_id} | Rep {rep}/{NUM_REPETITIONS} -> tokens: {res['token_usage']['completion_tokens']}, finish: {res['finish_reason']}")

    # 2. Top-p sweep (R2 primary as specified in Table: Max 17 C, rain 85%, landslide watch)
    print("\n[2/5] Top-p Sweep (0.3, 0.6, 0.9, 1.0)...")
    for top_p in [0.3, 0.6, 0.9, 1.0]:
        for input_id in ["R2", "R1"]:
            for rep in range(1, NUM_REPETITIONS + 1):
                sig = f"top_p_sweep_top_p_{top_p}_{input_id}_{rep}_None"
                if sig in existing_sigs:
                    continue
                res = client.generate_bulletin(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=f"Input readings: {TEST_INPUTS[input_id]}",
                    model=model,
                    temperature=DEFAULT_TEMP,
                    top_p=top_p,
                    max_tokens=DEFAULT_MAX_TOKENS,
                )
                log_and_save("top_p_sweep", "top_p", top_p, input_id, rep, None, res)
                print(f"  Top_p={top_p} | {input_id} | Rep {rep}/{NUM_REPETITIONS} -> tokens: {res['token_usage']['completion_tokens']}, finish: {res['finish_reason']}")

    # 3. Max tokens sweep (R3 primary as specified in Table: Max 29 C, rain 0%, heat)
    print("\n[3/5] Max Tokens Sweep (40, 80, 160)...")
    for max_tok in [40, 80, 160]:
        for input_id in ["R3", "R2"]:
            for rep in range(1, NUM_REPETITIONS + 1):
                sig = f"max_tokens_sweep_max_tokens_{max_tok}_{input_id}_{rep}_None"
                if sig in existing_sigs:
                    continue
                res = client.generate_bulletin(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=f"Input readings: {TEST_INPUTS[input_id]}",
                    model=model,
                    temperature=DEFAULT_TEMP,
                    top_p=DEFAULT_TOP_P,
                    max_tokens=max_tok,
                )
                log_and_save("max_tokens_sweep", "max_tokens", max_tok, input_id, rep, None, res)
                print(f"  MaxTokens={max_tok} | {input_id} | Rep {rep}/{NUM_REPETITIONS} -> tokens: {res['token_usage']['completion_tokens']}, finish: {res['finish_reason']}")

    # 4. Stop sequence sweep (R4 primary as specified in Table: Max 12 C, rain 60%, fog)
    print("\n[4/5] Stop Sequence Sweep (none, double_newline, advisory_colon)...")
    stop_configs = [
        ("none", None),
        ("double_newline", ["\n\n"]),
        ("advisory_colon", ["Advisory:"]),
    ]
    for stop_label, stop_seq in stop_configs:
        for input_id in ["R4"]:
            for rep in range(1, NUM_REPETITIONS + 1):
                sig = f"stop_sequence_sweep_stop_sequence_{stop_label}_{input_id}_{rep}_None"
                if sig in existing_sigs:
                    continue
                res = client.generate_bulletin(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=f"Input readings: {TEST_INPUTS[input_id]}",
                    model=model,
                    temperature=DEFAULT_TEMP,
                    top_p=DEFAULT_TOP_P,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    stop=stop_seq,
                )
                log_and_save("stop_sequence_sweep", "stop_sequence", stop_label, input_id, rep, None, res)
                print(f"  Stop={stop_label} | {input_id} | Rep {rep}/{NUM_REPETITIONS} -> tokens: {res['token_usage']['completion_tokens']}, finish: {res['finish_reason']}")

    # 5. Seed stability sweep (R5 primary as specified in Table: Max 21 C, rain 30%, advisory: none)
    print("\n[5/5] Seed Stability Sweep (Seed 42 vs Unseeded)...")
    for seed_val in [42, None]:
        seed_label = "fixed_42" if seed_val is not None else "unseeded"
        for input_id in ["R5"]:
            for rep in range(1, NUM_REPETITIONS + 1):
                sig = f"seed_stability_sweep_seed_{seed_label}_{input_id}_{rep}_{seed_val}"
                if sig in existing_sigs:
                    continue
                res = client.generate_bulletin(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=f"Input readings: {TEST_INPUTS[input_id]}",
                    model=model,
                    temperature=DEFAULT_TEMP,
                    top_p=DEFAULT_TOP_P,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    seed=seed_val,
                )
                log_and_save("seed_stability_sweep", "seed", seed_label, input_id, rep, seed_val, res)
                print(f"  Seed={seed_label} | {input_id} | Rep {rep}/{NUM_REPETITIONS} -> tokens: {res['token_usage']['completion_tokens']}")

    # Write full CSV
    if records:
        keys = list(records[0].keys())
        with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)

    print(f"\nAll experiments complete! Total logged runs: {len(records)}")
    print(f"CSV saved to: {raw_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma2:2b", help="Model name")
    args = parser.parse_args()
    run_all_experiments(model=args.model)
