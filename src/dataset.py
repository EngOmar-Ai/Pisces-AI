from datasets import load_dataset
from dotenv import load_dotenv

import sys

load_dotenv()

ENGLISH_WIKI_TRAIN_CONFIG_1 = {
    'start_index': 0,
    'end_index': 500000,
    'destination_filepath': "../data/EnglishWiki/Train/EnglishWikiTrainChunk1.txt"
}

ENGLISH_WIKI_TRAIN_CONFIG_2 = {
    'start_index': 500000,
    'end_index': 1000000,
    'destination_filepath': "../data/EnglishWiki/Train/EnglishWikiTrainChunk2.txt"
}

ENGLISH_WIKI_TRAIN_CONFIG_3 = {
    'start_index': 1000000,
    'end_index': 1500000,
    'destination_filepath': "../data/EnglishWiki/Train/EnglishWikiTrainChunk3.txt"

}

ENGLISH_WIKI_TRAIN_CONFIG_4 = {
    'start_index': 1500000,
    'end_index': 2000000,
    'destination_filepath': "../data/EnglishWiki/Train/EnglishWikiTrainChunk4.txt"
}

ENGLISH_WIKI_TRAIN_CONFIG_5 = {
    'start_index': 2000000,
    'end_index': 2500000,
    'destination_filepath': "../data/EnglishWiki/Train/EnglishWikiTrainChunk5.txt"
}

ENGLISH_WIKI_TRAIN_CONFIG_6 = {
    'start_index': 2500000,
    'end_index': 3000000,
    'destination_filepath': "../data/EnglishWiki/Train/EnglishWikiTrainChunk6.txt"
}

ENGLISH_WIKI_TRAIN_CONFIG_7 = {
    'start_index': 3000000,
    'end_index': 3500000,
    'destination_filepath': "../data/EnglishWiki/Train/EnglishWikiTrainChunk7.txt"
}

ENGLISH_WIKI_TRAIN_CONFIG_8 = {
    'start_index': 3500000,
    'end_index': 4000000,
    'destination_filepath': "../data/EnglishWiki/Train/EnglishWikiTrainChunk8.txt"
}

ENGLISH_WIKI_TRAIN_CONFIG_9 = {
    'start_index': 4000000,
    'end_index': 4500000,
    'destination_filepath': "../data/EnglishWiki/Train/EnglishWikiTrainChunk9.txt"
}

ENGLISH_WIKI_TRAIN_CONFIG_10 = {
    'start_index': 4500000,
    'end_index': 5000000,
    'destination_filepath': "../data/EnglishWiki/Train/EnglishWikiTrainChunk10.txt"
}

ENGLISH_WIKI_TEST_CONFIG_1 = {
    'start_index': 5000000,
    'end_index': 5500000,
    'destination_filepath': "../data/EnglishWiki/Test/EnglishWikiTestChunk1.txt"
}

ENGLISH_WIKI_TEST_CONFIG_2 =  {
    'start_index': 5500000,
    'end_index': 6000000,
    'destination_filepath': "../data/EnglishWiki/Test/EnglishWikiTestChunk2.txt"
}

def save_english_wikipedia_chunk(start_index: int, end_index: int, destination_filepath: str) -> None:
    """
    Saves a specified index range of the English Wikipedia dataset to a text file.

    Downloads a subset of the 'wikimedia/wikipedia' (20231101.en) dataset based on
    the provided start and end indices, formats each article with an end-of-sequence
    token (<eos>), and writes the output sequentially to the destination file.

    Args:
        start_index (int): The starting row index of the dataset slice (inclusive).
        end_index (int): The ending row index of the dataset slice (exclusive).
        destination_filepath (str): The file path where the text chunk will be saved.

    Raises:
        SystemExit: If the dataset fails to load or a fatal error occurs during execution,
            terminating the program with an error description.

    Side Effects:
        - Automatically creates missing parent directories for the destination file path.
        - Overwrites or creates a text file at `destination_filepath`.
        - Prints an execution report and itemized sample-level error logs to stdout.
    """

    logs, status = [], True

    try:
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split=f"train[{start_index}:{end_index}]")

        with open(destination_filepath, "w", encoding="utf-8") as file:
            for sample in dataset:
                try:
                    file.write(f"{sample['text']} <|endoftext|>\n\n")
                except Exception as error:
                    logs.append({
                        "id": sample.get("id"),
                        "url": sample.get("url"),
                        "title": sample.get("title"),
                        "error": str(error)
                    })
                    continue

    except Exception as error:
        status = False
        sys.exit(f"Failed To Save English Wikipedia Chunk,\nEncountered An Error While Loading The Dataset.\n\nError: {error}")

    finally:
        if status:
            print("========================================================")
            print("|                        Report                        |")
            print("========================================================")
            print()

            if logs:
                print(f"Failed To Save {len(logs)} Samples")
                print("\n-----\n")
                print(f"Logs: ")
                for log in logs:
                    print(f"    ID: {log['id']}, URL: {log['url']}, Title: {log['title']}, Error: {log['error']}")
            else:
                print(f"Successfully Saved English Wikipedia Chunk To {destination_filepath}, No Samples Skipped")

if __name__ == "__main__":
    ...