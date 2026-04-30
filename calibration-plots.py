# %%

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

CHOICES = ["A", "B", "C", "D"]


def load_data(path, key="forget_mcqa_prob"):
    with open(path, "r") as f:
        data = json.load(f)

    values = data[key]["value_by_index"]

    y_true = []
    y_pred = []
    probs = []

    for k in values:
        item = values[k]

        y_true.append(item["label"].strip())
        y_pred.append(item["generated choice"].strip())
        probs.append(item["prob"])

    probs = np.array(probs)

    return y_true, y_pred, probs


# -----------------------
# Accuracy + F1
# -----------------------

def compute_accuracy_f1(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1


# -----------------------
# Brier score (multiclass)
# -----------------------

def compute_brier(y_true, probs):

    y_true_idx = np.array([CHOICES.index(y) for y in y_true])

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y_true_idx)), y_true_idx] = 1

    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))

    return brier


# -----------------------
# ECE / MCE
# -----------------------

def compute_ece_mce(y_true, probs, n_bins=10):

    y_true_idx = np.array([CHOICES.index(y) for y in y_true])

    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)

    correct = (pred == y_true_idx).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):

        mask = (conf >= bins[i]) & (conf < bins[i + 1])

        if np.sum(mask) == 0:
            continue

        acc_bin = np.mean(correct[mask])
        conf_bin = np.mean(conf[mask])

        gap = abs(acc_bin - conf_bin)

        ece += np.sum(mask) / len(conf) * gap
        mce = max(mce, gap)

    return ece, mce

# -----------------------
# Reliability Diagram
# -----------------------


def plot_reliability_diagram(y_true, probs, n_bins=10, title="", key=""):

    y_true_idx = np.array([CHOICES.index(y) for y in y_true])

    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)

    correct = (pred == y_true_idx).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)

    bin_acc = []
    bin_conf = []
    bin_count = []

    for i in range(n_bins):

        mask = (conf >= bins[i]) & (conf < bins[i + 1])

        if np.sum(mask) == 0:
            bin_acc.append(0)
            bin_conf.append(0)
            bin_count.append(0)
            continue

        bin_acc.append(np.mean(correct[mask]))
        bin_conf.append(np.mean(conf[mask]))
        bin_count.append(np.sum(mask))

    bin_acc = np.array(bin_acc)
    bin_conf = np.array(bin_conf)
    bin_count = np.array(bin_count)

    # ----- plot -----

    # fig = plt.figure()

    # bars = accuracy
    plt.bar(
        np.linspace(0.05, 0.95, n_bins),
        bin_acc,
        width=0.08,
        alpha=0.6,
        label="Accuracy",
    )

    # # confidence line
    # plt.plot(
    #     np.linspace(0.05, 0.95, n_bins),
    #     bin_conf,
    #     marker="o",
    #     label="Confidence",
    # )

    # perfect calibration
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - {key}", fontsize=20)

    plt.legend()

    # return fig
    # plt.savefig(f"saves/eval/plots/phi-1_5/{title}_{key}_reliability_diagram.png", dpi=300, bbox_inches="tight")
    # plt.show()
    


# -----------------------
# MAIN
# -----------------------

def print_metrics_and_plot(path, title, subfig_nrows=None, subfig_ncols=None, subfig_pos=None):
    
    ret_values = []
    
    forget_percentage = int(title.split()[-1].replace("%",""))
    retain_percentage = 100 - forget_percentage
    title = " ".join(title.split()[:-1])
    
    for i, (key, key_title) in enumerate([
        ("retain_mcqa_prob", f"Retain {retain_percentage:2.0f}%"),
        ("forget_mcqa_prob", f"Forget {forget_percentage:2.0f} %") 
        ]):
        print(f"--- {key_title} ---")
        y_true, y_pred, probs = load_data(path, key=key)

        acc, f1 = compute_accuracy_f1(y_true, y_pred)
        brier = compute_brier(y_true, probs)
        ece, mce = compute_ece_mce(y_true, probs, n_bins=100)

        print("Accuracy:", acc)
        print("F1:", f1)
        print("Brier:", brier)
        print("ECE:", ece)
        print("MCE:", mce)
        
        if subfig_nrows is not None and subfig_ncols is not None and subfig_pos is not None:
            plt.subplot(subfig_nrows, subfig_ncols, subfig_pos+i)
        plot_reliability_diagram(y_true, probs, n_bins=10, title=title, key=key_title)
        
        ret_values.append({"key": key, "accuracy": acc, "f1": f1, "brier": brier, "ece": ece, "mce": mce})
        
    return ret_values

###############################################################################

# %% [markdown]
# # Phi


# %% [markdown]
# ## Reference models at all forget sizes

# %%
model_name = "phi-1_5"
path_template = "saves/eval/{task_name}/RELU_EVAL.json"
unlearn_path_template = "saves/unlearn/{task_name}/{checkpoint}/evals/RELU_EVAL.json"
tasks = [
    ("relu_{model_name}_pretrained_retain90_batch32", "Pretrained 10%"),
    ("relu_{model_name}_pretrained_retain95_batch32", "Pretrained 5%"),
    ("relu_{model_name}_pretrained_retain99_batch32", "Pretrained 1%"),

    ("relu_{model_name}_full_retain90", "Full Finetuned 10%"),
    ("relu_{model_name}_full_retain95", "Full Finetuned 5%"),
    ("relu_{model_name}_full_retain99", "Full Finetuned 1%"),


    ("relu_{model_name}_retain90_batch1", "Retained 10%"),
    ("relu_{model_name}_retain95_batch1", "Retained 5%"),
    ("relu_{model_name}_retain99_batch1", "Retained 1%"),
]

nrows = 3
ncols = 6
plt.figure(figsize=(ncols*5, nrows*4))

results = []

for i, (task, task_name) in enumerate(tasks):
    task = task.format(model_name=model_name)
    try:
        print(f"=== {task, i} ===")
        metrics = print_metrics_and_plot(path_template.format(task_name=task), title=task_name, subfig_nrows=3, subfig_ncols=6, subfig_pos=i*2+1)
        for m in metrics:
            results.append({"task": task, "checkpoint": 0, **m})
    except Exception as e:
        print(f"Error processing {task}: {e}")
        raise e
    
plt.tight_layout()
df = pd.DataFrame(results)
# store all the results in a JSON file
with open("saves/eval/calibration_results-phi_1_5-forgetall.json", "w") as f:
    json.dump(results, f, indent=4)


# %% [markdown]
# # OLD Llama reports

# %% [markdown]
# ## Llama 3 1B

# %%
model_name = "Llama-3.2-1B-Instruct"
path_template = "saves/eval/{task_name}/RELU_EVAL.json"
unlearn_path_template = "saves/unlearn/{task_name}/{checkpoint}/evals/RELU_EVAL.json"
tasks = [
    ("old_llama_evals/relu_{model_name}_pretrained_retain90", "Pretrained 10%"),
    ("old_llama_evals/relu_{model_name}_pretrained_retain95", "Pretrained 5%"),
    ("old_llama_evals/relu_{model_name}_pretrained_retain99", "Pretrained 1%"),

    ("old_llama_evals/relu_{model_name}_full_retain90", "Full Finetuned 10%"),
    ("old_llama_evals/relu_{model_name}_full_retain95", "Full Finetuned 5%"),
    ("old_llama_evals/relu_{model_name}_full_retain99", "Full Finetuned 1%"),


    ("old_llama_evals/relu_{model_name}_retain90", "Retained 10%"),
    ("old_llama_evals/relu_{model_name}_retain95", "Retained 5%"),
    ("old_llama_evals/relu_{model_name}_retain99", "Retained 1%"),
]

nrows = 3
ncols = 6
plt.figure(figsize=(ncols*5, nrows*4))

results = []

for i, (task, task_name) in enumerate(tasks):
    task = task.format(model_name=model_name)
    try:
        print(f"=== {task, i} ===")
        metrics = print_metrics_and_plot(path_template.format(task_name=task), title=task_name, subfig_nrows=3, subfig_ncols=6, subfig_pos=i*2+1)
        for m in metrics:
            results.append({"task": task, "checkpoint": 0, **m})
    except Exception as e:
        print(f"Error processing {task}: {e}")
        raise e
    
plt.tight_layout()
df = pd.DataFrame(results)
# store all the results in a JSON file
with open(f"saves/eval/calibration_results-{model_name}-forgetall.json", "w") as f:
    json.dump(results, f, indent=4)


# %% [markdown]
# ## Llama 3.2 3B

# %%
model_name = "Llama-3.2-3B-Instruct"
path_template = "saves/eval/{task_name}/RELU_EVAL.json"
unlearn_path_template = "saves/unlearn/{task_name}/{checkpoint}/evals/RELU_EVAL.json"
tasks = [
    ("old_llama_evals/relu_{model_name}_pretrained_retain90", "Pretrained 10%"),
    ("old_llama_evals/relu_{model_name}_pretrained_retain95", "Pretrained 5%"),
    ("old_llama_evals/relu_{model_name}_pretrained_retain99", "Pretrained 1%"),

    ("old_llama_evals/relu_{model_name}_full_retain90", "Full Finetuned 10%"),
    ("old_llama_evals/relu_{model_name}_full_retain95", "Full Finetuned 5%"),
    ("old_llama_evals/relu_{model_name}_full_retain99", "Full Finetuned 1%"),


    ("old_llama_evals/relu_{model_name}_retain90", "Retained 10%"),
    ("old_llama_evals/relu_{model_name}_retain95", "Retained 5%"),
    ("old_llama_evals/relu_{model_name}_retain99", "Retained 1%"),
]

nrows = 3
ncols = 6
plt.figure(figsize=(ncols*5, nrows*4))

results = []

for i, (task, task_name) in enumerate(tasks):
    task = task.format(model_name=model_name)
    try:
        print(f"=== {task, i} ===")
        metrics = print_metrics_and_plot(path_template.format(task_name=task), title=task_name, subfig_nrows=3, subfig_ncols=6, subfig_pos=i*2+1)
        for m in metrics:
            results.append({"task": task, "checkpoint": 0, **m})
    except Exception as e:
        print(f"Error processing {task}: {e}")
        raise e
    
plt.tight_layout()
df = pd.DataFrame(results)
# store all the results in a JSON file
with open(f"saves/eval/calibration_results-{model_name}-forgetall.json", "w") as f:
    json.dump(results, f, indent=4)


# %% [markdown]
# ## Llama 3 8B

# %%
model_name = "Llama-3.1-8B-Instruct"
path_template = "saves/eval/{task_name}/RELU_EVAL.json"
unlearn_path_template = "saves/unlearn/{task_name}/{checkpoint}/evals/RELU_EVAL.json"
tasks = [
    ("old_llama_evals/relu_{model_name}_pretrained_retain90", "Pretrained 10%"),
    ("old_llama_evals/relu_{model_name}_pretrained_retain95", "Pretrained 5%"),
    ("old_llama_evals/relu_{model_name}_pretrained_retain99", "Pretrained 1%"),

    ("old_llama_evals/relu_{model_name}_full_retain90", "Full Finetuned 10%"),
    ("old_llama_evals/relu_{model_name}_full_retain95", "Full Finetuned 5%"),
    ("old_llama_evals/relu_{model_name}_full_retain99", "Full Finetuned 1%"),


    ("old_llama_evals/relu_{model_name}_retain90", "Retained 10%"),
    ("old_llama_evals/relu_{model_name}_retain95", "Retained 5%"),
    ("old_llama_evals/relu_{model_name}_retain99", "Retained 1%"),
]

nrows = 3
ncols = 6
plt.figure(figsize=(ncols*5, nrows*4))

results = []

for i, (task, task_name) in enumerate(tasks):
    task = task.format(model_name=model_name)
    try:
        print(f"=== {task, i} ===")
        metrics = print_metrics_and_plot(path_template.format(task_name=task), title=task_name, subfig_nrows=3, subfig_ncols=6, subfig_pos=i*2+1)
        for m in metrics:
            results.append({"task": task, "checkpoint": 0, **m})
    except Exception as e:
        print(f"Error processing {task}: {e}")
        raise e
    
plt.tight_layout()
df = pd.DataFrame(results)
# store all the results in a JSON file
with open(f"saves/eval/calibration_results-{model_name}-forgetall.json", "w") as f:
    json.dump(results, f, indent=4)


# %% [markdown]
# # New Llama Reports

# %% [markdown]
# ## Llama 3.2 1B

# %%
model_name = "Llama-3.2-1B-Instruct"
path_template = "saves/eval/{task_name}/RELU_EVAL.json"
unlearn_path_template = "saves/unlearn/{task_name}/{checkpoint}/evals/RELU_EVAL.json"
tasks = [
    ("relu_{model_name}_pretrained_retain90_batch32", "Pretrained 10%"),
    ("relu_{model_name}_pretrained_retain95_batch32", "Pretrained 5%"),
    ("relu_{model_name}_pretrained_retain99_batch32", "Pretrained 1%"),

    ("relu_{model_name}_full_retain90_batch32", "Full Finetuned 10%"),
    ("relu_{model_name}_full_retain95_batch32", "Full Finetuned 5%"),
    ("relu_{model_name}_full_retain99_batch32", "Full Finetuned 1%"),


    ("relu_{model_name}_retain90_batch32", "Retained 10%"),
    ("relu_{model_name}_retain95_batch32", "Retained 5%"),
    ("relu_{model_name}_retain99_batch32", "Retained 1%"),
]

nrows = 3
ncols = 6
plt.figure(figsize=(ncols*5, nrows*4))

results = []

for i, (task, task_name) in enumerate(tasks):
    task = task.format(model_name=model_name)
    try:
        print(f"=== {task, i} ===")
        metrics = print_metrics_and_plot(path_template.format(task_name=task), title=task_name, subfig_nrows=3, subfig_ncols=6, subfig_pos=i*2+1)
        for m in metrics:
            results.append({"task": task, "checkpoint": 0, **m})
    except Exception as e:
        print(f"Error processing {task}: {e}")
        # raise e
    
plt.tight_layout()
df = pd.DataFrame(results)
# store all the results in a JSON file
with open(f"saves/eval/calibration_results-{model_name}-forgetall.json", "w") as f:
    json.dump(results, f, indent=4)


# %% [markdown]
# ## Llama 3.2 3B

# %%
model_name = "Llama-3.2-3B-Instruct"
path_template = "saves/eval/{task_name}/RELU_EVAL.json"
unlearn_path_template = "saves/unlearn/{task_name}/{checkpoint}/evals/RELU_EVAL.json"
tasks = [
    ("relu_{model_name}_pretrained_retain90_batch32", "Pretrained 10%"),
    ("relu_{model_name}_pretrained_retain95_batch32", "Pretrained 5%"),
    ("relu_{model_name}_pretrained_retain99_batch32", "Pretrained 1%"),

    ("relu_{model_name}_full_retain90_batch32", "Full Finetuned 10%"),
    ("relu_{model_name}_full_retain95_batch32", "Full Finetuned 5%"),
    ("relu_{model_name}_full_retain99_batch32", "Full Finetuned 1%"),


    ("relu_{model_name}_retain90_batch32", "Retained 10%"),
    ("relu_{model_name}_retain95_batch32", "Retained 5%"),
    ("relu_{model_name}_retain99_batch32", "Retained 1%"),
]

nrows = 3
ncols = 6
plt.figure(figsize=(ncols*5, nrows*4))

results = []

for i, (task, task_name) in enumerate(tasks):
    task = task.format(model_name=model_name)
    try:
        print(f"=== {task, i} ===")
        metrics = print_metrics_and_plot(path_template.format(task_name=task), title=task_name, subfig_nrows=3, subfig_ncols=6, subfig_pos=i*2+1)
        for m in metrics:
            results.append({"task": task, "checkpoint": 0, **m})
    except Exception as e:
        print(f"Error processing {task}: {e}")
        # raise e
    
plt.tight_layout()
df = pd.DataFrame(results)
# store all the results in a JSON file
with open(f"saves/eval/calibration_results-{model_name}-forgetall.json", "w") as f:
    json.dump(results, f, indent=4)


# %% [markdown]
# ## Llama 3.1 8B

# %%
model_name = "Llama-3.1-8B-Instruct"
path_template = "saves/eval/{task_name}/RELU_EVAL.json"
unlearn_path_template = "saves/unlearn/{task_name}/{checkpoint}/evals/RELU_EVAL.json"
tasks = [
    ("relu_{model_name}_pretrained_retain90_batch32", "Pretrained 10%"),
    ("relu_{model_name}_pretrained_retain95_batch32", "Pretrained 5%"),
    ("relu_{model_name}_pretrained_retain99_batch32", "Pretrained 1%"),

    ("relu_{model_name}_full_retain90_batch32", "Full Finetuned 10%"),
    ("relu_{model_name}_full_retain95_batch32", "Full Finetuned 5%"),
    ("relu_{model_name}_full_retain99_batch32", "Full Finetuned 1%"),


    ("relu_{model_name}_retain90_batch32", "Retained 10%"),
    ("relu_{model_name}_retain95_batch32", "Retained 5%"),
    ("relu_{model_name}_retain99_batch32", "Retained 1%"),
]

nrows = 3
ncols = 6
plt.figure(figsize=(ncols*5, nrows*4))

results = []

for i, (task, task_name) in enumerate(tasks):
    task = task.format(model_name=model_name)
    try:
        print(f"=== {task, i} ===")
        metrics = print_metrics_and_plot(path_template.format(task_name=task), title=task_name, subfig_nrows=3, subfig_ncols=6, subfig_pos=i*2+1)
        for m in metrics:
            results.append({"task": task, "checkpoint": 0, **m})
    except Exception as e:
        print(f"Error processing {task}: {e}")
        # raise e
    
plt.tight_layout()
df = pd.DataFrame(results)
# store all the results in a JSON file
with open(f"saves/eval/calibration_results-{model_name}-forgetall.json", "w") as f:
    json.dump(results, f, indent=4)



