from datasets import load_dataset
import os

def download_openwebtext(folder="data"):
    """
    Downloads the openwebtext dataset and saves it as a text file.
    """
    # Create the folder in the parent directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(os.path.dirname(script_dir), folder)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    filepath = os.path.join(data_folder, "openwebtext.txt")

    print("Downloading openwebtext dataset...")
    try:
        dataset = load_dataset("Skylion007/openwebtext", split="train")
        with open(filepath, "w", encoding="utf-8") as f:
            for example in dataset:
                f.write(example["text"])
        print(f"Dataset saved to {filepath}")
        return filepath
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    download_openwebtext()
