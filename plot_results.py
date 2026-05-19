import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    base_dir = "/DATAX/divyaksh/Projects/unlearning/unlearning-calibration/saves/unlearn"
    data = []

    # Iterate through all run directories
    for run_dir in glob.glob(os.path.join(base_dir, "tofu_*")):
        dir_name = os.path.basename(run_dir)
        parts = dir_name.split("_")
        if len(parts) >= 4:
            # Format: tofu_{model}_{forget_size}_{algorithm}
            model = parts[1]
            forget_size = parts[2]
            algorithm = "_".join(parts[3:])
            
            # Find all checkpoint directories
            for ckpt_dir in glob.glob(os.path.join(run_dir, "checkpoint-*")):
                ckpt_name = os.path.basename(ckpt_dir)
                try:
                    checkpoint = int(ckpt_name.split("-")[1])
                except ValueError:
                    continue
                    
                summary_path = os.path.join(ckpt_dir, "evals", "TOFU_SUMMARY.json")
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, "r") as f:
                            summary = json.load(f)
                        data.append({
                            "model": model,
                            "forget_size": forget_size,
                            "algorithm": algorithm,
                            "checkpoint": checkpoint,
                            "mia_min_k": summary.get("mia_min_k"),
                            "mia_min_k_plus_plus": summary.get("mia_min_k_plus_plus"),
                            "privleak": summary.get("privleak")
                        })
                    except Exception as e:
                        print(f"Error reading {summary_path}: {e}")

    df = pd.DataFrame(data)

    if df.empty:
        print("No data found!")
        return

    # Sort the dataframe for better readability
    df = df.sort_values(by=["model", "forget_size", "algorithm", "checkpoint"])
    print(f"Successfully loaded {len(df)} records.")
    
    # Save the aggregated data to a CSV
    csv_path = os.path.join(base_dir, "aggregated_unlearning_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved aggregated dataframe to {csv_path}")

    # Plotting
    metrics = ["mia_min_k", "mia_min_k_plus_plus", "privleak"]
    models = sorted(df['model'].unique())
    forget_sizes = sorted(df['forget_size'].unique())

    sns.set_theme(style="whitegrid")

    for metric in metrics:
        # Create a grid of subplots
        fig, axes = plt.subplots(nrows=len(forget_sizes), ncols=len(models), 
                                 figsize=(6 * len(models), 5 * len(forget_sizes)))
        
        # Normalize axes to 2D array
        if len(forget_sizes) == 1 and len(models) == 1:
            axes = [[axes]]
        elif len(forget_sizes) == 1:
            axes = [axes]
        elif len(models) == 1:
            axes = [[ax] for ax in axes]
            
        for i, f_size in enumerate(forget_sizes):
            for j, mod in enumerate(models):
                ax = axes[i][j]
                subset = df[(df['forget_size'] == f_size) & (df['model'] == mod)]
                
                if not subset.empty:
                    sns.lineplot(data=subset, x="checkpoint", y=metric, hue="algorithm", 
                                 marker="o", ax=ax, errorbar=None)
                
                ax.set_title(f"{mod}\n{f_size}")
                
                if j == 0:
                    ax.set_ylabel(metric)
                else:
                    ax.set_ylabel("")
                    
                ax.set_xlabel("Checkpoint")
                    
                # Manage legends: only put one legend outside the last plot
                if i == 0 and j == len(models) - 1:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Algorithm")
                else:
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()
                            
        plt.tight_layout()
        output_path = os.path.join(base_dir, f"{metric}_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
