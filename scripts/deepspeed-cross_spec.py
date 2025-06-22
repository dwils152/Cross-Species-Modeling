import argparse
import datetime
import torch
import numpy as np
import pandas as pd
from transformers import TrainingArguments, Trainer
from SequenceDataset import SequenceDataset
from HFDatasetWrapper import HFDatasetWrapper
from transformers import AutoModelForTokenClassification
from CRMTokenClassifier import CRMTokenClassifier
from sklearn.metrics import accuracy_score
from peft import PeftModel, PeftConfig
import sys

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(np.float32)
    preds_flat = preds.reshape(-1)
    labels_flat = labels.reshape(-1)
    acc = accuracy_score(labels_flat, preds_flat)
    return {"accuracy": acc}

def main(args):
    # Recreate your dataset (must be identical to training)
    dataset = SequenceDataset(
        args.fasta,
        args.labels,
        args.model_name,
        args.max_length,
    )
    hf_dataset = HFDatasetWrapper(dataset)

    # Load your fine-tuned model
    model = CRMTokenClassifier(args.model_checkpoint)
    lora_model = PeftModel.from_pretrained(model, args.lora_dir)

    # Set up a Trainer for prediction
    training_args = TrainingArguments(
        output_dir=args.run_name,
        per_device_eval_batch_size=4,
        report_to=[],
    )
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        compute_metrics=None if args.labels is None else compute_metrics,
    )

    # Run predictions
    predictions = trainer.predict(hf_dataset)
    logits = predictions.predictions
    probs = 1 / (1 + np.exp(-logits))

    # Flatten the outputs
    flat_probs = probs.flatten()

    # Create a DataFrame and save to CSV
    df = pd.DataFrame({
        "probs": flat_probs,
    })
    df.to_csv(args.output_csv, index=False)

    print("Test metrics:", predictions.metrics)
    print(f"Saved all predictions and labels to '{args.output_csv}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prediction script')
    parser.add_argument('--fasta', type=str, required=True,
                        help='Path to the dataset (FASTA format).')
    parser.add_argument('--labels', type=str, required=False, default=None,
                        help="Optional path to the labels file (omit for pure prediction).")
    parser.add_argument('--model_name', type=str,
                        default='InstaDeepAI/nucleotide-transformer-v2-250m-multi-species',
                        help='Pre-trained model name or path.')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Maximum sequence length for the inputs.')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Path to the fine-tuned model checkpoint.')
    parser.add_argument('--lora_dir',type=str, default='Lora',
                        help='Path to the Lora adapter and config')
    parser.add_argument('--output_csv', type=str, default='predictions.csv',
                        help='Output CSV file to save predictions and labels.')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name for the run (defaults to YYYYMMDD_HHMMSS)')
    parser.add_argument("--deepspeed", type=str, default=None, 
                    help="Path to the DeepSpeed config file.")
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank for distributed inference/training."
    )
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    main(args)
