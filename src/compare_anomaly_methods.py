
from pathlib import Path

import pandas as pd


RESULTS_PATH = Path(
    "outputs/isolation_forest/isolation_forest_results.csv"
)
OUTPUT_DIR = Path("outputs/anomaly_comparison")


def main():
    df = pd.read_csv(RESULTS_PATH)

    # Recalculate the transparent, formula-based rule.
    required = [
        "Feed_Flow_m3h",
        "Recovery_Pct",
        "Water_Production_m3h",
    ]

    checkable = df[required].notna().all(axis=1)

    expected = (
        df["Feed_Flow_m3h"] * df["Recovery_Pct"] / 100
    )

    absolute_difference = (
        df["Water_Production_m3h"] - expected
    ).abs()

    df["Rule_Checked"] = checkable
    df["Rule_Flag"] = (
        checkable & (absolute_difference > 1.0)
    )

    df["ML_Flag"] = (
        df["IF_Status"] == "Review: ML anomaly"
    )

    # Compare only rows that can be checked by the rule.
    comparable = df.loc[checkable].copy()

    rule_only = comparable[
        comparable["Rule_Flag"] & ~comparable["ML_Flag"]
    ]
    ml_only = comparable[
        ~comparable["Rule_Flag"] & comparable["ML_Flag"]
    ]
    both = comparable[
        comparable["Rule_Flag"] & comparable["ML_Flag"]
    ]

    neither = comparable[
        ~comparable["Rule_Flag"] & ~comparable["ML_Flag"]
    ]

    print("=== RULE-BASED VS ISOLATION FOREST ===")
    print(f"Total dataset rows: {len(df):,}")
    print(f"Rows checkable by rule: {len(comparable):,}")
    print(f"Rule-based flags: {int(comparable['Rule_Flag'].sum()):,}")
    print(f"ML flags in comparable rows: {int(comparable['ML_Flag'].sum()):,}")
    print(f"Flagged by both: {len(both):,}")
    print(f"Rule only: {len(rule_only):,}")
    print(f"ML only: {len(ml_only):,}")
    print(f"Neither: {len(neither):,}")

    union_count = len(
        comparable[
            comparable["Rule_Flag"] | comparable["ML_Flag"]
        ]
    )
    intersection_count = len(both)

    if union_count:
        print(
            "Jaccard overlap:",
            round(intersection_count / union_count, 4),
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    both.to_csv(OUTPUT_DIR / "flagged_by_both.csv", index=False)
    rule_only.to_csv(OUTPUT_DIR / "rule_only.csv", index=False)
    ml_only.to_csv(OUTPUT_DIR / "ml_only.csv", index=False)

    print(f"\nComparison files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
