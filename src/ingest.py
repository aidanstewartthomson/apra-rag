import polars as pl
from pathlib import Path


def load_documents(directory: Path) -> list[pl.DataFrame]:
    documents = []

    for file in directory.glob("*.csv"):
        header = pl.read_csv(file, has_header=False, n_rows=1)
        title = header[0, 1]

        df = pl.read_csv(file, skip_lines=3)
        df = df.with_columns(pl.lit(title).alias("title"))
        documents.append(df)

    return documents


def normalise_documents(documents: list[pl.DataFrame]) -> pl.DataFrame:
    frames = []

    for df in documents:
        frame = (
            df.with_row_index("idx")
            .drop("Fragment ID")
            .rename({"Heading": "section", "Content": "text"})
            .with_columns(pl.col(["section", "text"]).replace("", None))
            .with_columns(pl.col("section").forward_fill())
            .filter(pl.col("text").is_not_null())
            .group_by(["title", "section"])
            .agg(pl.col("text").sort_by("idx").str.join(" "))
        )
        frames.append(frame)

    return pl.concat(frames)


def main():
    directory = Path("data/raw")

    documents = load_documents(directory)
    documents = normalise_documents(documents)

    print(documents.head())


if __name__ == "__main__":
    main()
