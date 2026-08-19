from model import SEQUENCE_LENGTH, BATCH_SIZE, GRAD_CLIP_VALUE, model, device, criterion, scheduler, optimizer, metrics, path
from data import load_english_wiki_batch

from math import ceil, exp
import sys

import mysql.connector
import torch

from torch.nn.utils import clip_grad_norm_ as clip_gradients

def train(samples: int, database: dict, training_table: str, validation_table: str):
    connection, cursor = None, None

    try:
        connection = mysql.connector.connect(**database)
        cursor = connection.cursor()

        load()

        model.train()

        training_loss = 0

        for counter in range(samples):
            x, y = load_english_wiki_batch(cursor, training_table, BATCH_SIZE)
            x, y = x.to(device), y.to(device)

            logits = model(x).transpose(1, 2)

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()

            clip_gradients(model.parameters(), GRAD_CLIP_VALUE)

            optimizer.step()
            scheduler.step()

            training_loss = training_loss + loss.item()

        training_loss = training_loss / samples
        training_perplexity = exp(training_loss)

        validation_loss = validate(cursor, ceil(samples * 0.1), validation_table)
        validation_perplexity = exp(validation_loss)

        tokens_seen = SEQUENCE_LENGTH * samples

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

        choice = input("Would you like to save the model? (y/n): ").strip().lower()
        if choice == 'y' or choice == 'yes':
            save()
            print(f"Saved The Model Successfully At {path}")
        else:
            print(f"Model Discarded...")

    except mysql.connector.Error as error:
        sys.exit(f"Error while connecting to MySQL: {error}")

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

def validate(cursor, samples: int, validation_table: str):
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