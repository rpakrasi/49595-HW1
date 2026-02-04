import json
import random
import re
from typing import List, Dict

ASSISTANT_SPEAKER = "TRUMP"  # BIDEN or TRUMP

SYSTEM_PROMPT = "You are Joe Biden and you are debating Donald Trump."
if ASSISTANT_SPEAKER == "TRUMP":
    SYSTEM_PROMPT = "You are Donald Trump and you are debating Joe Biden."


def clean_text(text):
    """Remove anything inside square brackets and extra whitespace."""
    # Remove [ ... ] including brackets
    cleaned = re.sub(r"\[.*?\]", "", text)
    # Collapse multiple spaces and strip
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def transcript_to_jsonl(input_txt, output_jsonl):
    turns = []

    current_speaker = None
    current_text = []

    # Step 1: parse transcript into (speaker, text) turns
    with open(input_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()

            # Match SPEAKER:
            match = re.match(r"^([A-Z][A-Z\s]+):\s*(.*)", line)

            if match:
                # Save previous turn
                if current_speaker and current_text:
                    turns.append(
                        (current_speaker, " ".join(current_text).strip())
                    )

                current_speaker = match.group(1).strip()
                current_text = [match.group(2).strip()]
            else:
                # Continuation of the same speaker (multiline)
                if current_speaker:
                    current_text.append(line.strip())

        # Save last turn
        if current_speaker and current_text:
            turns.append(
                (current_speaker, " ".join(current_text).strip())
            )

    # Step 2: build JSONL from turns
    included = 0
    skipped = 0
    with open(output_jsonl, "w", encoding="utf-8") as out:
        for i, (speaker, text) in enumerate(turns):
            if speaker == ASSISTANT_SPEAKER and i > 0:
                prev_speaker, prev_text = turns[i - 1]

                # Clean both user and assistant text
                prev_text_clean = clean_text(prev_text)
                text_clean = clean_text(text)

                # Skip if assistant response is too short
                if len(text_clean) < 3:
                    skipped += 1
                    continue

                conversation = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prev_text_clean},
                        {"role": "assistant", "content": text_clean}
                    ]
                }

                out.write(json.dumps(conversation, ensure_ascii=False) + "\n")
                included += 1

    print(f"Saved {included} conversations to {output_jsonl}, skipped {skipped} short responses.")


def split_jsonl_dataset(
        input_file: str,
        train_file: str,
        valid_file: str,
        train_ratio: float = 0.8,
        seed: int = 42
) -> None:
    """
    Splits a JSONL dataset into training and validation JSONL files.

    Args:
        input_file (str): Path to the input JSONL file (each line a JSON object with 'prompt' and 'completion').
        train_file (str): Path to save the training JSONL file.
        valid_file (str): Path to save the validation JSONL file.
        train_ratio (float, optional): Proportion of data to use for training. Defaults to 0.8.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """
    # Load JSONL dataset line by line
    data: List[Dict] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                data.append(json.loads(line))

    # Shuffle data
    random.seed(seed)
    random.shuffle(data)

    # Split data
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    valid_data = data[split_idx:]

    # Write JSONL helper
    def write_jsonl(filename: str, dataset: List[Dict]):
        with open(filename, "w", encoding="utf-8") as f:
            for entry in dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Save splits
    write_jsonl(train_file, train_data)
    write_jsonl(valid_file, valid_data)

    print(f"Dataset split complete!")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(valid_data)}")


if __name__ == "__main__":
    data_path = "../data"
    output_folder = "biden-responses"
    if ASSISTANT_SPEAKER == "TRUMP":
        output_folder = "trump-responses"

    # input_file = f"{data_path}/raw-data/trump_biden_1.txt"
    # output_file = f"{data_path}/{output_folder}/trump_biden_1.jsonl"
    # transcript_to_jsonl(input_file, output_file)

    jsonl_file = f"{data_path}/{output_folder}/combined.jsonl"
    split_jsonl_dataset(jsonl_file, f"{data_path}/{output_folder}/training.jsonl",
                        f"{data_path}/{output_folder}/validation.jsonl", train_ratio=0.8)
