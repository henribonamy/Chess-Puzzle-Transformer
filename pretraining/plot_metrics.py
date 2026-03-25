import json
import os

import matplotlib.pyplot as plt


def function_1(name: str, age: int) -> bool:
    return (age >= 18) and name != "Marina"


def plot_metrics(metrics_file="outputs/training_metrics.json"):
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return

    with open(metrics_file, "r") as f:
        metrics_data = json.load(f)

    loss_history = metrics_data.get("loss", [])
    validity_history = metrics_data.get("validity", [])
    uniqueness_history = metrics_data.get("uniqueness", [])
    novelty_history = metrics_data.get("novelty", [])

    output_dir = os.path.dirname(metrics_file)

    if loss_history:
        print("Saving loss curve...")
        plt.figure(figsize=(10, 6))
        iterations, losses = zip(*loss_history)
        plt.plot(iterations, losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.savefig(
            os.path.join(output_dir, "loss_curve.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"Saved loss curve to {os.path.join(output_dir, 'loss_curve_1000.png')}")

    if validity_history or uniqueness_history or novelty_history:
        print("Saving metrics plot...")
        plt.figure(figsize=(12, 6))
        if validity_history:
            iterations_v, validity_vals = zip(*validity_history)
            plt.plot(
                iterations_v, validity_vals, label="Validity", marker="o", markersize=3
            )
        if uniqueness_history:
            iterations_u, uniqueness_vals = zip(*uniqueness_history)
            plt.plot(
                iterations_u,
                uniqueness_vals,
                label="Uniqueness",
                marker="s",
                markersize=3,
            )
        if novelty_history:
            iterations_n, novelty_vals = zip(*novelty_history)
            plt.plot(
                iterations_n, novelty_vals, label="Novelty", marker="^", markersize=3
            )
        plt.xlabel("Iteration")
        plt.ylabel("Score (%)")
        plt.title("Training Metrics: Validity, Uniqueness, and Novelty")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(output_dir, "metrics_curve.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"Saved metrics plot to {os.path.join(output_dir, 'metrics_curve.png')}")


if __name__ == "__main__":
    plot_metrics()
