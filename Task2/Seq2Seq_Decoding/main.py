import torch
import json
import time
import argparse
from functools import partial
from torch.utils.data import DataLoader
from pathlib import Path
import os
import sys

# Optional: Import wandb for logging (only if enabled)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Dynamically set the base directory and add it to sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR  / "Task2" / "Seq2Seq_Decoding" ))

from Task2.RuleClassification.logger import logger
from Task2.Seq2Seq_Decoding.helpers import load_data, load_sankrit_dictionary, save_task2_predictions
from Task2.Seq2Seq_Decoding.vocabulary import make_vocabulary, PAD_TOKEN
from Task2.Seq2Seq_Decoding.dataset import index_dataset, collate_fn, eval_collate_fn
from Task2.Seq2Seq_Decoding.model import build_model, build_optimizer, save_model
from Task2.Seq2Seq_Decoding.training import train
from Task2.Seq2Seq_Decoding.evaluate import (
    evaluate_model,
    print_metrics,
    format_predictions,
    convert_eval_if_translit,
)
from Task2.Seq2Seq_Decoding.scoring import evaluate
from Task2.Seq2Seq_Decoding.extract_rules import get_token_rule_mapping, get_rules

# Ignore warning (who cares?)
import warnings

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate rule classification model")
    parser.add_argument("--config", type=str, default="config.cfg", help="Path to config file")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load config file
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(config_path, "r") as cfg:
        config = json.load(cfg)

    translit = config.get("translit", False)
    test = config.get("test", False)
    tune = config.get("tune", False)

    # Set project root and update config paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    config["cwd"] = project_root
    config["train_path"] = os.path.join(project_root, "sanskrit", "wsmp_train.json")
    config["eval_path"] = os.path.join(project_root, "sanskrit", "corrected_wsmp_dev.json")
    config["test_path"] = os.path.join(project_root, "sanskrit")

    # Initialize Weights & Biases if enabled
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project="rule-classifier", config=config, name=config.get("name"))

    # Load training and evaluation data
    logger.info("Load data")
    train_data = load_data(config["train_path"], translit)
    eval_data = load_data(config["eval_path"], translit)
    logger.info(f"Loaded {len(train_data)} train sents")
    logger.info(f"Loaded {len(eval_data)} test sents")

    # Exctract rules
    print("\nExtracting rules")
    rules = get_rules(train_data, use_tag=True)
    print(f"Extracted {len(rules)} rules")

    # Make vocabulary
    print("\nMake vocab")
    # TODO: Include UNK chars and labels
    vocabulary, tag_encoder, char2index, index2char = make_vocabulary(train_data)
    pad_tag = tag_encoder.transform([PAD_TOKEN]).item()

    # Index data
    print("\nIndex train data")
    train_data_indexed, train_dataset, discarded = index_dataset(
        train_data, char2index, tag_encoder
    )
    print(f"\nDiscarded {discarded} sentences from train data")
    print("Index eval data")
    eval_data_indexed, eval_dataset, discarded = index_dataset(
        eval_data, char2index, tag_encoder, eval=True
    )
    print(f"\nDiscarded {discarded} sentences from eval data")

    # Build dataloaders
    batch_size = config["batch_size"]
    collate_fn = partial(collate_fn, pad_tag=pad_tag)
    eval_collate_fn = partial(eval_collate_fn, pad_tag=pad_tag)

    train_dataloader = DataLoader(
        train_data_indexed, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_data_indexed,
        batch_size=batch_size,
        collate_fn=eval_collate_fn,
        shuffle=False,
    )

    # Build model
    print("\nBuild model")
    model = build_model(config, vocabulary, tag_encoder)
    use_cuda = config["cuda"]
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using devide: {device}")

    model = model.to(device)

    # Build optimizer
    optimizer = build_optimizer(model, config)

    # Train
    print("\nTrain\n")
    epochs = config["epochs"]
    print(f"Training for {epochs} epochs\n")

    start = time.time()

    model, optimizer = train(
        model, optimizer, train_dataloader, epochs, device, pad_tag
    )

    # Save model
    name = config["name"]
    save_model(model, optimizer, vocabulary, tag_encoder, char2index, index2char, name)

    # Evaluate
    mode = config["eval_mode"]
    # dictionary_path = config.get("dictionary_path", None)
    # if dictionary_path is None:
    #    dictionary = None
    # else:
    #    dictionary = load_sankrit_dictionary(dictionary_path)

    eval_predictions = evaluate_model(
        model,
        eval_dataloader,
        tag_encoder,
        char2index,
        index2char,
        device,
        mode=mode,
        rules=rules,
    )
    formatted_predictions = format_predictions(eval_predictions, translit=translit)
    # print_metrics(eval_predictions, eval_dataset)

    duration = time.time() - start
    print(f"\nDuration: {duration:.2f} seconds.\n")

    # Create submission
    print("\nCreate submission files")
    save_task2_predictions(formatted_predictions, duration)

    # Evaluate
    print("\nEvaluating")
    # print_metrics(eval_predictions, eval_dataset)
    # Task 2 Evaluation
    if translit:
        eval_data = convert_eval_if_translit(eval_data)
    scores = evaluate([dp[1] for dp in eval_data], formatted_predictions, task_id="t2")
