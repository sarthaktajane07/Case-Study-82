import os
import json
import re
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes exact Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def normalized_pairwise_edit_distance(texts: List[str]) -> float:
    """Computes average normalized pairwise edit distance across texts."""
    if len(texts) <= 1:
        return 0.0
    distances = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            max_len = max(len(texts[i]), len(texts[j]))
            if max_len == 0:
                distances.append(0.0)
            else:
                dist = levenshtein_distance(texts[i], texts[j]) / max_len
                distances.append(dist)
    return float(np.mean(distances)) if distances else 0.0

def distinct_ngrams_ratio(texts: List[str], n: int = 1) -> float:
    """Computes Distinct-n ratio (unique n-grams / total n-grams) across a list of texts."""
    all_ngrams = []
    for text in texts:
        tokens = re.findall(r"\b\w+\b", text.lower())
        if len(tokens) >= n:
            ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return float(len(set(all_ngrams)) / len(all_ngrams))

# Ground truth keywords & entities for faithfulness auditing
HAZARD_HALLUCINATION_TERMS = [
    r"road\s+clos(?:ed|ure)",
    r"highway\s+clos(?:ed|ure)",
    r"bridge\s+wash(?:ed|out)",
    r"power\s+outage",
    r"electricity\s+cut",
    r"tree\s+fall",
    r"evacuat(?:e|ion)",
    r"flash\s+flood",
    r"torrential",
    r"severe\s+blizzard",
    r"thunderstorm",
    r"hailstorm",
    r"snowfall",
    r"dangerously\s+high\s+winds",
]

def detect_unsupported_facts(response_text: str, input_id: str, input_text: str) -> Tuple[int, List[str]]:
    """
    Identifies facts mentioned in the bulletin that are absent from the input readings.
    Returns (count_of_invented_facts, list_of_invented_facts).
    """
    text_lower = response_text.lower()
    invented = []

    # Check for hallucinated severe hazards
    for term in HAZARD_HALLUCINATION_TERMS:
        match = re.search(term, text_lower)
        if match:
            # Check if this hazard is justified by the input advisory
            if "landslide" in match.group(0) and "landslide watch" in input_text.lower():
                continue
            invented.append(match.group(0))

    # Check advisory hallucination: if input says 'advisory: none', model shouldn't invent an advisory
    if "advisory: none" in input_text.lower():
        if re.search(r"\b(warning|severe alert|stay indoors|danger alert)\b", text_lower):
            invented.append("spurious emergency alert on advisory:none")

    # Check precipitation hallucination: R3 has rain 0%, R1 has rain 10%
    if "rain 0%" in input_text.lower():
        if re.search(r"\b(showers|rainy|downpour|rainfall expected)\b", text_lower):
            invented.append("invented rainfall when rain is 0%")
    elif "rain 10%" in input_text.lower():
        if re.search(r"\b(heavy rain|continuous rain|downpour)\b", text_lower):
            invented.append("exaggerated heavy rain when rain is 10%")

    # Check wind hallucination: R3 has wind 5 km/h, R1 has wind 8 km/h
    if "wind 5 km/h" in input_text.lower() or "wind 8 km/h" in input_text.lower():
        if re.search(r"\b(strong winds|gale|blustery|gusty winds|high winds)\b", text_lower):
            invented.append("invented strong wind when wind is <10 km/h")

    # Check temperature distortion
    if "max 29 c" in input_text.lower():
        if re.search(r"\b(freezing|chilly|cold|frost)\b", text_lower):
            invented.append("cold weather mentioned for 29 C")
    elif "max 12 c" in input_text.lower():
        if re.search(r"\b(hot|heatwave|warm summer)\b", text_lower):
            invented.append("hot weather mentioned for 12 C")

    return len(invented), invented

def evaluate_experiments(
    raw_csv_path: str = "/Users/sarthak/Downloads/Case_Study_82/data/experiments_raw.csv",
    output_dir: str = "/Users/sarthak/Downloads/Case_Study_82/data",
):
    df = pd.read_csv(raw_csv_path)
    print(f"Loaded {len(df)} experiment records from {raw_csv_path}")

    # Compute word count
    df["word_count"] = df["response_text"].apply(lambda x: len(str(x).split()))
    df["char_count"] = df["response_text"].apply(lambda x: len(str(x)))

    # Aggregated metrics table
    summary_rows = []

    grouped = df.groupby(["experiment_type", "param_varied", "param_value", "input_id"])
    for (exp_type, param_varied, param_val, input_id), group in grouped:
        texts = group["response_text"].tolist()
        pairwise_edit = normalized_pairwise_edit_distance(texts)
        dist_1 = distinct_ngrams_ratio(texts, n=1)
        dist_2 = distinct_ngrams_ratio(texts, n=2)
        mean_tokens = group["completion_tokens"].mean()
        std_tokens = group["completion_tokens"].std()
        mean_words = group["word_count"].mean()
        truncation_rate = (group["finish_reason"] == "length").mean()

        summary_rows.append({
            "experiment_type": exp_type,
            "param_varied": param_varied,
            "param_val": param_val,
            "input_id": input_id,
            "sample_size": len(group),
            "pairwise_edit_distance": round(pairwise_edit, 4),
            "distinct_1": round(dist_1, 4),
            "distinct_2": round(dist_2, 4),
            "mean_tokens": round(mean_tokens, 2),
            "std_tokens": round(0.0 if np.isnan(std_tokens) else std_tokens, 2),
            "mean_words": round(mean_words, 2),
            "truncation_rate": round(truncation_rate, 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary metrics to {summary_path}")

    # Faithfulness Sample of 20 outputs across settings
    print("\nAuditing Faithfulness across 20 representative outputs...")
    sample_indices = []
    
    # Stratified selection:
    # 8 from temperature sweep (temp 0.0, 0.3, 0.7, 1.0, 1.4)
    temp_df = df[df["experiment_type"] == "temperature_sweep"]
    for temp in [0.0, 0.3, 0.7, 1.0, 1.4]:
        sub = temp_df[temp_df["temperature"] == temp]
        if not sub.empty:
            sample_indices.extend(sub.sample(min(2, len(sub)), random_state=42).index.tolist())

    # 4 from top_p sweep (0.3, 0.6, 0.9, 1.0)
    topp_df = df[df["experiment_type"] == "top_p_sweep"]
    for p in [0.3, 0.6, 0.9, 1.0]:
        sub = topp_df[topp_df["top_p"] == p]
        if not sub.empty:
            sample_indices.extend(sub.sample(1, random_state=42).index.tolist())

    # 4 from max_tokens sweep (40, 80, 160)
    tok_df = df[df["experiment_type"] == "max_tokens_sweep"]
    for tok in [40, 80, 160]:
        sub = tok_df[tok_df["max_tokens"] == tok]
        if not sub.empty:
            sample_indices.extend(sub.sample(1, random_state=42).index.tolist())

    # Remaining from stop sequence & seed
    other_df = df[~df.index.isin(sample_indices)]
    needed = 20 - len(set(sample_indices))
    if needed > 0 and not other_df.empty:
        sample_indices.extend(other_df.sample(needed, random_state=42).index.tolist())

    sample_20_df = df.loc[list(dict.fromkeys(sample_indices))[:20]].copy()

    faithfulness_rows = []
    for idx, row in sample_20_df.iterrows():
        inv_count, inv_list = detect_unsupported_facts(
            str(row["response_text"]), str(row["input_id"]), str(row["input_text"])
        )
        faithfulness_rows.append({
            "run_index": idx,
            "experiment_type": row["experiment_type"],
            "input_id": row["input_id"],
            "input_text": row["input_text"],
            "temperature": row["temperature"],
            "top_p": row["top_p"],
            "max_tokens": row["max_tokens"],
            "finish_reason": row["finish_reason"],
            "response_text": str(row["response_text"]).replace("\n", " "),
            "invented_facts_count": inv_count,
            "invented_facts_details": "; ".join(inv_list) if inv_list else "None (Faithful)",
            "is_faithful": (inv_count == 0),
        })

    faithfulness_df = pd.DataFrame(faithfulness_rows)
    faithfulness_path = os.path.join(output_dir, "faithfulness_sample_20.csv")
    faithfulness_df.to_csv(faithfulness_path, index=False)
    print(f"Saved 20-sample faithfulness evaluation to {faithfulness_path}")
    print(f"Faithful outputs: {(faithfulness_df['is_faithful']).sum()} / 20 ({(faithfulness_df['is_faithful']).mean()*100:.1f}%)")

if __name__ == "__main__":
    evaluate_experiments()
