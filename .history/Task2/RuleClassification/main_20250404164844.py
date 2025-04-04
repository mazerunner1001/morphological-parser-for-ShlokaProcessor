import json
import time
import torch
import pprint
import warnings
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader

import sys
sys.path.append("c:/Users/spran/OneDrive - Indian Institute of Technology (BHU), Varanasi/Desktop/BTP/TueSan/")
sys.path.append("c:/Users/spran/OneDrive - Indian Institute of Technology (BHU), Varanasi/Desktop/BTP/TueSan/Task2/RuleClassification")
sys.path.append("c:/Users/spran/OneDrive - Indian Institute of Technology (BHU), Varanasi/Desktop/BTP/TueSan/Task2/Seq2Seq_Decoding")

from Task2.RuleClassification.logger import logger
from Task2.RuleClassification.training import train
from Task2.RuleClassification.scoring import evaluate
from Task2.RuleClassification.stemming_rules import evaluate_coverage
from Task2.RuleClassification.generate_dataset import construct_train_dataset
from Task2.RuleClassification.model import build_model, build_optimizer, save_model
from Task2.RuleClassification.uni2intern import internal_transliteration_to_unicode as to_uni
from Task2.RuleClassification.index_dataset import index_dataset, train_collate_fn, eval_collate_fn
from Task2.RuleClassification.helpers import load_data, save_task2_predictions, load_task2_test_data
from Task2.RuleClassification.evaluate import (
    evaluate_model,
    print_metrics,
    format_predictions,
    convert_eval_if_translit,
)

# Ignore warning (who cares?)

warnings.filterwarnings("ignore")
pp = pprint.PrettyPrinter(indent=4)


if __name__ == "__main__":
    with open("/config.cfg") as cfg:
        config = json.load(cfg)

    # Read booleans
    translit = config["translit"]
    test = config["test"]
    tune = config["tune"]

    # Load data
    logger.info("Load data")
    train_data = load_data(config["train_path"], translit)
    if not test:
        eval_data = load_data(config["eval_path"], translit)
    else:
        eval_data = load_task2_test_data(
            Path(config["test_path"], "task_2_input_sentences.tsv"), translit
        )

    logger.info(f"Loaded {len(train_data)} train sents")
    logger.info(f"Loaded {len(eval_data)} test sents")

    # logger.debug(eval_data[0])

    # Generate datasets
    logger.info("Generate training dataset")
    tag_rules = config["tag_rules"]
    stemming_rule_cutoff = config["stemming_rule_cutoff"]
    train_data, stem_rules, tags, discarded = construct_train_dataset(
        train_data, tag_rules, stemming_rule_cutoff
    )
    logger.info(f"Training data contains {len(train_data)} sents")
    logger.info(f"Discarded {discarded} invalid sents from train data")
    logger.info(f"Collected {len(stem_rules)} Stemming rules")
    logger.info(f"Collected {len(tags)} morphological tags")

    if tag_rules:
        logger.info("Stemming rules contain morphological tag")
    else:
        logger.info("Morphological tags are predicted separately from stems")

    if not test:
        evaluate_coverage(eval_data, stem_rules, logger, tag_rules)

    logger.info("Index dataset")

    # Build vocabulary and index the dataset
    indexed_train_data, indexed_eval_data, indexer = index_dataset(
        train_data, eval_data, stem_rules, tags, tag_rules
    )

    logger.info(f"{len(indexer.vocabulary)} chars in vocab:\n{indexer.vocabulary}\n")

    # Build dataloaders
    logger.info("Build training dataloader")
    batch_size = config["batch_size"]
    train_dataloader = DataLoader(
        indexed_train_data,
        batch_size=batch_size,
        collate_fn=train_collate_fn,
        shuffle=True,
    )

    # Build model
    logger.info("Build model")
    model = build_model(config, indexer, tag_rules)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_parameters} trainable parameters")

    use_cuda = config["cuda"]
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using devide: {device}")

    model = model.to(device)

    # Build optimizer
    logger.info("Build optimizer")
    optimizer = build_optimizer(model, config)

    # Train
    epochs = config["epochs"]
    logger.info(f"Training for {epochs} epochs\n")

    start = time.time()

    model, optimizer = train(
        model, optimizer, train_dataloader, epochs, device, tag_rules
    )

    # Save model
    logger.info("Saving model")
    name = config["name"]
    save_model(
        model, optimizer, indexer, stem_rules, tags, name,
    )

    # Prediction
    logger.info("Predicting")
    eval_dataloader = DataLoader(
        indexed_eval_data,
        batch_size=batch_size,
        collate_fn=eval_collate_fn,
        shuffle=False,
    )
    eval_predictions = evaluate_model(
        model, eval_dataloader, indexer, device, tag_rules, translit
    )

    duration = time.time() - start
    logger.info(f"Duration: {duration:.2f} seconds.\n")

    # Create submission
    logger.info("Create submission files")
    save_task2_predictions(eval_predictions, duration)

    # Evaluation
    if not test:
        logger.info("Evaluating")
        # print_metrics(eval_predictions, eval_dataset)
        # Task 2 Evaluation
        if translit:
            eval_data = convert_eval_if_translit(eval_data)
        scores = evaluate([dp[1] for dp in eval_data], eval_predictions, task_id="t2")
