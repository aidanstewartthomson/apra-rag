import polars as pl
from pathlib import Path


def load_documents(directory: Path) -> list[pl.DataFrame]:
    documents = []

    for file in directory.glob("*.csv"):
        df = pl.read_csv(file, skip_lines=3)
        documents.append(df)

    return documents


def main():
    directory = Path("data/raw")
    documents = load_documents(directory)

    for df in documents:
        print(df.head())


if __name__ == "__main__":
    main()
