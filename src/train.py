from model import SEQUENCE_LENGTH, BATCH_SIZE, GRAD_CLIP_VALUE, model, device, criterion, scheduler, optimizer, metrics, path
from data import load_english_wiki_batch

from math import exp
import sys

import mysql.connector
import torch

from torch.nn.utils import clip_grad_norm_ as clip_gradients

def train(database: dict, training_samples: int, validation_samples: int, training_table: str, validation_table: str):
    """
    Run a training loop for a fixed number of batches, then validate and checkpoint.

    Connects to the MySQL database, loads any existing checkpoint, trains the
    model for `samples` batches (backprop + gradient clipping + optimizer/scheduler
    step per batch),

    runs a validation pass on the validation batches specified, logs a
    metrics report to stdout, and saves an updated checkpoint before closing
    the database connection.

    Args:
        database: Keyword args passed to mysql.connector.connect (host, user, password, database, etc.).
        training_samples: Number of training batches to run this call.
        validation_samples: Number of validation batches to run this call.
        training_table: Name of the MySQL table to pull training batches from.
        validation_table: Name of the MySQL table to pull validation batches from.

    Raises:
        SystemExit: If the MySQL connection fails.
    """

    connection, cursor = None, None

    try:
        connection = mysql.connector.connect(**database)
        cursor = connection.cursor()

        load()

        model.train()

        training_loss = 0

        for counter in range(training_samples):
            x, y = load_english_wiki_batch(cursor, training_table, BATCH_SIZE)
            x, y = x.to(device), y.to(device)

            logits = model(x).transpose(1, 2)

            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            clip_gradients(model.parameters(), GRAD_CLIP_VALUE)

            optimizer.step()
            scheduler.step()

            training_loss = training_loss + loss.item()

        training_loss = training_loss / training_samples
        training_perplexity = exp(training_loss)

        validation_loss = validate(cursor, validation_samples, validation_table)
        validation_perplexity = exp(validation_loss)

        tokens_seen = BATCH_SIZE * SEQUENCE_LENGTH * training_samples

        metrics['tokens_seen'].append(tokens_seen)
        metrics['training_loss'].append(training_loss)
        metrics['validation_loss'].append(validation_loss)
        metrics['training_perplexity'].append(training_perplexity)
        metrics['validation_perplexity'].append(validation_perplexity)

        print(f"------------------------------------------- REPORT --------------------------------------------")
        print(f"| Training Loss | Training Perplexity | Validation Loss | Validation Perplexity | Tokens Seen |")
        print(f"| ------------- | ------------------- | --------------- | --------------------- | ----------- |")
        print(f"| {training_loss:^13.5f} | {training_perplexity:^19.5f} | {validation_loss:^15.5f} | {validation_perplexity:^21.5f} | {tokens_seen:^11} |")
        print(f"-----------------------------------------------------------------------------------------------")

        save()

    except mysql.connector.Error as error:
        sys.exit(f"Error while connecting to MySQL: {error}")

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

def validate(cursor, samples: int, validation_table: str):
    """
    Run a no-grad evaluation pass and return the mean validation loss.

    Switches the model to eval mode, runs `samples` batches from
    `validation_table` without gradient tracking, averages the loss, then
    switches the model back to train mode before returning.

    Args:
       cursor: Open MySQL cursor used to fetch validation batches.
       samples: Number of validation batches to run.
       validation_table: Name of the MySQL table to pull validation batches from.

    Returns:
       float: Mean validation loss across the sampled batches.
    """

    model.eval()

    validation_loss = 0

    with torch.no_grad():
        for counter in range(samples):
            x, y = load_english_wiki_batch(cursor, validation_table, BATCH_SIZE)
            x, y = x.to(device), y.to(device)

            logits = model(x).transpose(1, 2)

            loss = criterion(logits, y)

            validation_loss = validation_loss + loss.item()

    validation_loss = validation_loss / samples

    model.train()

    return validation_loss

def save():
    """
    Save a checkpoint containing model/optimizer/scheduler state and metrics.

    Writes model_state_dict, optimizer_state_dict, scheduler_state_dict, and
    the full metrics history (tokens seen, training/validation loss and
    perplexity) to `path` via torch.save.
    """

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),

        'tokens_seen': metrics['tokens_seen'],

        'training_loss': metrics['training_loss'],
        'validation_loss': metrics['validation_loss'],

        'training_perplexity': metrics['training_perplexity'],
        'validation_perplexity': metrics['validation_perplexity'],
    }
    torch.save(checkpoint, path)

def load():
    """
    Restore model/optimizer/scheduler state and metrics from a checkpoint.

    If a checkpoint exists at `path`, loads it (mapped to `device`) and
    restores model_state_dict, optimizer_state_dict, scheduler_state_dict,
    and the metrics history in place. If no checkpoint is found, leaves
    the current (default) state untouched and prints a notice.
    """

    if path.exists():
        data = torch.load(path, map_location=device)

        model.load_state_dict(data['model_state_dict'])
        optimizer.load_state_dict(data['optimizer_state_dict'])
        scheduler.load_state_dict(data['scheduler_state_dict'])

        metrics["tokens_seen"] = data['tokens_seen']

        metrics["training_loss"] = data['training_loss']
        metrics["validation_loss"] = data['validation_loss']

        metrics["training_perplexity"] = data['training_perplexity']
        metrics["validation_perplexity"] = data['validation_perplexity']
    else:
        print(f"No checkpoint found at {path}, Initializing Default Values")

if __name__ == "__main__":
    ...