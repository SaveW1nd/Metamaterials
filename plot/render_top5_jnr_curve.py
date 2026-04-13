from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "server_results" / "multiseed20_noscale_20260412" / "top5_jnr_curve_data.json"
OUTPUT_PATH = PROJECT_ROOT / "plot" / "jnr_top5_curve.pdf"


def load_curve_data(path: Path = DATA_PATH) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "top5_seeds" not in payload or "per_jnr" not in payload:
        raise ValueError("Curve data must contain 'top5_seeds' and 'per_jnr'.")
    points = payload["per_jnr"]
    if len(points) != 31:
        raise ValueError(f"Expected 31 JNR points, got {len(points)}.")
    jnr_values = [float(row["jnr_db"]) for row in points]
    if jnr_values != sorted(jnr_values):
        raise ValueError("JNR points must be sorted in ascending order.")
    return payload


def render_curve_pdf(
    data_path: Path = DATA_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = load_curve_data(data_path)
    points = payload["per_jnr"]

    jnr = [row["jnr_db"] for row in points]
    series = [
        ("$A_{T_l}$", [row["slice_width_hit_rate_mean"] * 100 for row in points], "#1f77b4", "o"),
        ("$A_{T_s}$", [row["sampling_interval_hit_rate_mean"] * 100 for row in points], "#ff7f0e", "s"),
        ("$A_x$", [row["modulation_floor_hit_rate_mean"] * 100 for row in points], "#2ca02c", "^"),
        ("$A_{total}$", [row["joint_hit_rate_mean"] * 100 for row in points], "#d62728", "D"),
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for label, values, color, marker in series:
        ax.plot(
            jnr,
            values,
            label=label,
            color=color,
            linewidth=2.1,
            marker=marker,
            markersize=5.8,
            markeredgewidth=0.7,
        )

    ax.set_xlabel("JNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(-10, 20)
    ax.set_ylim(0, 102)
    ax.set_xticks([-10, -5, 0, 5, 10, 15, 20])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    path = render_curve_pdf()
    print(path)
