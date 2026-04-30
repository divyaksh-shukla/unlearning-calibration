# convert RELU_EVAL.json to a format compatible with plot_reliability.py

import json
from pathlib import Path
from argparse import ArgumentParser


def convert_relu_to_plot_reliability_format(input_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    output_path = Path(input_path).parent
    forget_path = output_path / "forget-mcqa.json"
    retain_path = output_path / "retain-mcqa.json"

    forget_output = []
    forget_data = data['forget_mcqa_prob']
    for _, s in forget_data['value_by_index'].items():
        forget_output.append({
            "predicted_label": s['label'],
            "label": s['generated choice'],
            "option_logits": s['logit'] if 'logit' in s else None,
            "option_probs": s['prob'] if 'prob' in s else None,
        })
    with open(forget_path, 'w') as f:
        json.dump(forget_output, f, indent=2)
    
    retain_output = []
    retain_data = data['retain_mcqa_prob']
    for _, s in retain_data['value_by_index'].items():
        retain_output.append({
            "predicted_label": s['label'],
            "label": s['generated choice'],
            "option_logits": s['logit'] if 'logit' in s else None,
            "option_probs": s['prob'] if 'prob' in s else None,
        })
    with open(retain_path, 'w') as f:
        json.dump(retain_output, f, indent=2)

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--input_path", type=str, default="RELU_EVAL.json", help="Path to the input RELU_EVAL.json file")
    args = args.parse_args()
    input_path = args.input_path  # Path to the input RELU_EVAL.json file
    convert_relu_to_plot_reliability_format(input_path)

# USAGE:
"""
python convert_to_relu_output.py --input_path saves/eval/relu_phi-1_5_retain90_batch1_debug/RELU_EVAL.json
"""




