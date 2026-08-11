# ====================== #
#         Imports        #
# ====================== #

from dotenv import load_dotenv
from model import tokenizer

import numpy
import mysql.connector

import sys

# =============================== #
#        Load '.env' File         #
# =============================== #

load_dotenv()

# =============================================== #
#        Saving Text File To The Database         #
# =============================================== #

def save_text_file_to_database(filepath: str, database_configuration: dict, table: str, sequence_length: int, batch_size: int):
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
            counter += 1

            if counter == sequence_length + 1:

                array = numpy.array(sample, dtype=numpy.uint16).tobytes()

                batch.append((array,))

                counter = 0
                sample.clear()

                samples_count += 1

                if len(batch) == batch_size:

                    cursor.executemany(query, batch)
                    connection.commit()

                    print(f"Saved {samples_count:,} samples")

                    batch.clear()


    except mysql.connector.Error as error:
        if connection:
            connection.rollback()

        sys.exit(f"Failed To Connect To The Database: {error}")

    finally:
        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()

# ==================================================== #
#        Load Training Batch From The Database         #
# ==================================================== #

def load_english_wiki_train_batch():
    ...

# =================================================== #
#        Load Testing Batch From The Database         #
# =================================================== #

def load_english_wiki_test_batch():
    ...

# ============================= #
#             Main              #
# ============================= #

if __name__ == '__main__':
    ...

# db_config = {
#     "host": os.getenv("DB_HOST"),
#     "user": os.getenv("DB_USER"),
#     "password": os.getenv("DB_PASSWORD"),
#     "database": os.getenv("DB_NAME"),
# }
#
# db_table = "EnglishWikiTrainSamples"
#
# fp = r'../data/EnglishWiki/Train/EnglishWikiTrainChunk1'
# import time
# start = time.perf_counter()
# save_text_file_to_database(filepath=fp, database_configuration=db_config, table=db_table, sequence_length=512, batch_size=1000)
# end = time.perf_counter()
# print(f"Time Taken: {end-start}")
#
# tokens = numpy.frombuffer(binary, dtype=numpy.uint16).tolist()
