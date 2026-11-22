import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

def generate_plots(
    data_dir: str = "/Users/sarthak/Downloads/Case_Study_82/data",
    figures_dir: str = "/Users/sarthak/Downloads/Case_Study_82/figures",
):
    os.makedirs(figures_dir, exist_ok=True)
    raw_df = pd.read_csv(os.path.join(data_dir, "experiments_raw.csv"))
    summary_df = pd.read_csv(os.path.join(data_dir, "summary_metrics.csv"))
    faithfulness_df = pd.read_csv(os.path.join(data_dir, "faithfulness_sample_20.csv"))

    print("Generating Figure 1: Diversity vs Temperature & Top-p...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Diversity vs Temperature
    temp_summary = summary_df[summary_df["param_varied"] == "temperature"].copy()
    temp_summary["param_val"] = temp_summary["param_val"].astype(float)
    temp_grouped = temp_summary.groupby("param_val")[["pairwise_edit_distance", "distinct_1", "distinct_2"]].mean().reset_index()

    axes[0].plot(temp_grouped["param_val"], temp_grouped["pairwise_edit_distance"], marker="o", linewidth=2.5, label="Pairwise Edit Distance (Levenshtein)", color="#1f77b4")
    axes[0].plot(temp_grouped["param_val"], temp_grouped["distinct_2"], marker="s", linewidth=2, linestyle="--", label="Distinct-2 Gram Ratio", color="#2ca02c")
    axes[0].plot(temp_grouped["param_val"], temp_grouped["distinct_1"], marker="^", linewidth=1.5, linestyle=":", label="Distinct-1 Gram Ratio", color="#ff7f0e")
    axes[0].set_xlabel("Temperature ($T$)")
    axes[0].set_ylabel("Diversity Metric Value")
    axes[0].set_title("Lexical Diversity vs Temperature")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="upper left")
    axes[0].axvspan(0.3, 0.7, color="green", alpha=0.12, label="Acceptable Diversity Zone")

    # Diversity vs Top-p
    topp_summary = summary_df[summary_df["param_varied"] == "top_p"].copy()
    topp_summary["param_val"] = topp_summary["param_val"].astype(float)
    topp_grouped = topp_summary.groupby("param_val")[["pairwise_edit_distance", "distinct_1", "distinct_2"]].mean().reset_index()

    axes[1].plot(topp_grouped["param_val"], topp_grouped["pairwise_edit_distance"], marker="o", linewidth=2.5, label="Pairwise Edit Distance", color="#9467bd")
    axes[1].plot(topp_grouped["param_val"], topp_grouped["distinct_2"], marker="s", linewidth=2, linestyle="--", label="Distinct-2 Gram Ratio", color="#8c564b")
    axes[1].plot(topp_grouped["param_val"], topp_grouped["distinct_1"], marker="^", linewidth=1.5, linestyle=":", label="Distinct-1 Gram Ratio", color="#e377c2")
    axes[1].set_xlabel("Nucleus Sampling (Top-p)")
    axes[1].set_ylabel("Diversity Metric Value")
    axes[1].set_title("Lexical Diversity vs Top-p (at $T=0.7$)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper left")
    axes[1].axvspan(0.6, 0.9, color="green", alpha=0.12)

    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "fig1_diversity_vs_temperature_topp.png")
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved: {fig1_path}")

    print("Generating Figure 2: Faithfulness / Hallucination vs Temperature...")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Analyze faithfulness sample
    faith_temp = faithfulness_df[faithfulness_df["experiment_type"] == "temperature_sweep"].copy()
    if not faith_temp.empty:
        temp_faith_summary = faith_temp.groupby("temperature")["invented_facts_count"].mean().reset_index()
    else:
        # Fallback if sampled across types
        temp_faith_summary = faithfulness_df.groupby("temperature")["invented_facts_count"].mean().reset_index()

    # Overlay with diversity curve to show Pareto tradeoff
    ax1.bar(
        temp_faith_summary["temperature"].astype(str),
        temp_faith_summary["invented_facts_count"],
        width=0.4,
        color="#d62728",
        alpha=0.7,
        label="Avg Invented Facts per Bulletin",
    )
    ax1.set_xlabel("Sampling Temperature ($T$)")
    ax1.set_ylabel("Hallucination Count (Unsupported Facts)", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_ylim(0, max(temp_faith_summary["invented_facts_count"].max() + 1, 3))

    ax2 = ax1.twinx()
    ax2.plot(
        temp_grouped["param_val"].astype(str),
        temp_grouped["pairwise_edit_distance"],
        color="#1f77b4",
        marker="o",
        linewidth=2.5,
        label="Pairwise Edit Distance (Diversity)",
    )
    ax2.set_ylabel("Diversity (Pairwise Edit Distance)", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(False)

    plt.title("Faithfulness vs Diversity Tradeoff Across Temperatures")
    fig.tight_layout()
    fig2_path = os.path.join(figures_dir, "fig2_faithfulness_vs_temperature.png")
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved: {fig2_path}")

    print("Generating Figure 3: Length Statistics & Truncation vs Max Tokens...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    maxtok_df = raw_df[raw_df["experiment_type"] == "max_tokens_sweep"].copy()
    maxtok_summary = summary_df[summary_df["param_varied"] == "max_tokens"].copy()
    maxtok_summary["param_val"] = maxtok_summary["param_val"].astype(int)
    maxtok_agg = maxtok_summary.groupby("param_val")[["mean_tokens", "std_tokens", "truncation_rate"]].mean().reset_index()

    # Token Completion vs Limit
    axes[0].bar(
        maxtok_agg["param_val"].astype(str),
        maxtok_agg["mean_tokens"],
        yerr=maxtok_agg["std_tokens"],
        capsize=5,
        color="#3b528b",
        alpha=0.8,
        label="Generated Tokens (Mean ± Std)",
    )
    # Plot target budget reference lines
    for i, tok_val in enumerate(maxtok_agg["param_val"]):
        axes[0].axhline(tok_val, color="red", linestyle=":", alpha=0.5, label=f"Max Limit {tok_val}" if i == 0 else "")
    axes[0].set_xlabel("Configured max_tokens Setting")
    axes[0].set_ylabel("Completion Tokens Generated")
    axes[0].set_title("Generated Output Length vs Configured Limit")
    axes[0].legend(loc="upper left")

    # Finish Reason breakdown
    finish_breakdown = maxtok_df.groupby(["max_tokens", "finish_reason"]).size().unstack(fill_value=0)
    finish_breakdown_pct = finish_breakdown.div(finish_breakdown.sum(axis=1), axis=0) * 100
    finish_breakdown_pct.plot(
        kind="bar",
        stacked=True,
        ax=axes[1],
        color={"stop": "#2ca02c", "length": "#d62728", "error": "#7f7f7f"},
        alpha=0.85,
    )
    axes[1].set_xlabel("Configured max_tokens Setting")
    axes[1].set_ylabel("Percentage of Bulletins (%)")
    axes[1].set_title("Termination Reason ('stop' vs Truncation 'length')")
    axes[1].legend(title="Finish Reason", loc="center right")
    axes[1].set_xticklabels(finish_breakdown_pct.index, rotation=0)

    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "fig3_length_and_truncation_vs_maxtokens.png")
    plt.savefig(fig3_path)
    plt.close()
    print(f"Saved: {fig3_path}")

    print("Generating Figure 4: Stop Sequences and Seed Reproducibility...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Stop sequences effect on tokens
    stop_summary = summary_df[summary_df["param_varied"] == "stop_sequence"].copy()
    stop_agg = stop_summary.groupby("param_val")[["mean_tokens", "mean_words"]].mean().reset_index()

    axes[0].bar(
        stop_agg["param_val"],
        stop_agg["mean_tokens"],
        color=["#440154", "#21918c", "#fde725"],
        alpha=0.85,
    )
    axes[0].set_xlabel("Stop Sequence Applied")
    axes[0].set_ylabel("Average Completion Tokens")
    axes[0].set_title("Impact of Stop Sequences on Output Truncation")
    for i, v in enumerate(stop_agg["mean_tokens"]):
        axes[0].text(i, v + 2, f"{v:.1f}", ha="center", fontweight="bold")

    # Seed reproducibility
    seed_summary = summary_df[summary_df["param_varied"] == "seed"].copy()
    seed_agg = seed_summary.groupby("param_val")[["pairwise_edit_distance", "distinct_1"]].mean().reset_index()

    x = np.arange(len(seed_agg))
    width = 0.35
    axes[1].bar(x - width/2, seed_agg["pairwise_edit_distance"], width, label="Pairwise Edit Distance", color="#31688e")
    axes[1].bar(x + width/2, seed_agg["distinct_1"], width, label="Distinct-1 Ratio", color="#35b779")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(seed_agg["param_val"])
    axes[1].set_ylabel("Diversity Metric Value")
    axes[1].set_title("Reproducibility: Fixed Seed (42) vs Stochastic Unseeded")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "fig4_stop_sequences_and_seed_stability.png")
    plt.savefig(fig4_path)
    plt.close()
    print(f"Saved: {fig4_path}")

if __name__ == "__main__":
    generate_plots()
