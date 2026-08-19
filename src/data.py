from dotenv import load_dotenv
from model import tokenizer

import numpy
import mysql.connector

import random
import sys

load_dotenv()

def save_text_file_to_database(filepath: str, database_configuration: dict, table: str, sequence_length: int, batch_size: int) -> None:
    """
    Tokenize a text file and save fixed-length token sequences to MySQL.

    The input file must contain articles separated by the literal '<eos>\\n'
    marker. Each article is tokenized as a whole and followed by the
    tokenizer's EOS token ID. The resulting token stream is divided into
    non-overlapping samples containing `sequence_length + 1` tokens.

    Each sample is stored as a NumPy uint16 array converted to raw bytes and
    inserted into the specified MySQL table as a BLOB. Samples are inserted
    and committed in batches to improve database performance.

    Args:
        filepath: Path to the text file containing the articles.
        database_configuration: MySQL connection configuration passed to
            mysql.connector.connect().
        table: Name of the MySQL table into which samples are inserted.
        sequence_length: Number of input tokens in each training sample.
            Each stored sample therefore contains `sequence_length + 1`
            tokens, where tokens[:-1] are the inputs and tokens[1:] are
            the corresponding target tokens.
        batch_size: Number of samples to accumulate before inserting and
            committing them to the database.

    Raises:
        ValueError: If the tokenizer does not contain the expected EOS token.

    Note:
        Any remaining tokens that do not form a complete sample are discarded.
    """

    def extract_tokens(extraction_filepath: str):
        """
        Extracts token IDs line by line from a text file using a tokenizer.

        Args:
            extraction_filepath (str): The path to the text file to be read and tokenized.

        Yields:
            int: Individual token IDs generated from the encoded text lines.
        """

        with open(extraction_filepath, 'r', encoding="utf-8") as file:
            for line in file:
                tokens = tokenizer.encode(line).ids
                for token in tokens:
                    yield token

    connection = None
    cursor = None

    try:
        connection = mysql.connector.connect(**database_configuration)
        cursor = connection.cursor()

        sample, batch = [], []
        samples_count, counter = 0, 0

        query = f"INSERT INTO {table} (sample) VALUES (%s)"

        for t in extract_tokens(filepath):

            sample.append(t)

            if len(sample) == sequence_length + 1:

                array = numpy.array(sample, dtype=numpy.uint16).tobytes()
                batch.append((array,))

                sample.clear()
                samples_count += 1

                if len(batch) == batch_size:

                    cursor.executemany(query, batch)
                    connection.commit()

                    batch.clear()

                    print(f"Saved {samples_count:,} samples")

    except mysql.connector.Error as error:
        if connection:
            connection.rollback()

        sys.exit(f"Failed To Connect To The Database: {error}")

    finally:
        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()

def load_english_wiki_batch(cursor, table: str , batch_size: int) -> tuple:
    """
    Loads a random batch of training/testing data from a database table.

    This function retrieves a specified number of random records from the given
    database table, handles primary key gaps gracefully, and parses the binary
    sample data into NumPy arrays suitable for model training.

    Args:
      cursor: The database cursor object used to execute queries.
      table (str): The name of the database table to query.
      batch_size (int): The number of samples to load in the batch.

    Returns:
      tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
          - batch_input: Array of input tokens of shape (batch_size, 512).
          - batch_output: Array of target tokens of shape (batch_size, 512).
    """

    cursor.execute(f"SELECT MIN(id), MAX(id) FROM {table}")
    min_id, max_id = cursor.fetchone()

    if not min_id or not max_id:
        raise ValueError("The Table Is Empty")

    ids = [random.randint(min_id, max_id) for _ in range(batch_size)]
    placeholders = ','.join(['%s'] * batch_size)

    cursor.execute(f"SELECT sample FROM {table} WHERE id IN ({placeholders});", ids)

    samples = [row[0] for row in cursor.fetchall() if row[0] is not None]

    while len(samples) < batch_size:
        random_id = random.randint(min_id, max_id)

        cursor.execute(f"SELECT sample FROM {table} WHERE id = %s;", (random_id,))
        sample = cursor.fetchone()

        if sample is not None:
            samples.append(sample[0])

    batch_input, batch_output = [], []
    for sample in samples:
        tokens = numpy.frombuffer(sample, dtype=numpy.uint16).tolist()

        batch_input.append(tokens[:512])
        batch_output.append(tokens[1:])

    batch_input = numpy.array(batch_input, dtype=numpy.uint16)
    batch_output = numpy.array(batch_output, dtype=numpy.uint16)

    return batch_input, batch_output

if __name__ == '__main__':
    ...