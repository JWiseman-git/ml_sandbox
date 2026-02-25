"""Movie Genre Classification from Character Metadata using Bag-of-Words.

Dataset: CMU Movie Summary Corpus
  - movie.metadata.tsv  : movie-level metadata including Freebase genre labels
  - character.metadata.tsv : character names keyed to movies

Pipeline
--------
1. Download and extract the CMU corpus tarball.
2. Load both TSV files, assign column names, log shapes at every cleaning step.
3. Parse the Freebase JSON genre column; extract the primary (first) genre.
4. Normalise genres and character names; drop rows that become invalid.
5. Group all character names per movie into a single text string.
6. Merge character text with genre labels.
7. Vectorise with three strategies: CountVectorizer (count), CountVectorizer
   (binary), and TfidfVectorizer.
8. Train a LogisticRegression classifier for each vectoriser; print accuracy.
9. Print a full classification report for the best-performing variant.
"""

import ast
import os
import tarfile
import urllib.request

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_URL = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TARBALL = os.path.join(DATA_DIR, "MovieSummaries.tar.gz")
EXTRACTED = os.path.join(DATA_DIR, "MovieSummaries")

# Column names from the CMU README schema (TSVs ship without headers).
MOVIE_METADATA_COLS = [
    "wikipedia_id",       # 0 – Wikipedia article ID
    "freebase_id",        # 1 – Freebase movie ID
    "title",              # 2 – Movie title
    "release_date",       # 3 – Release date (string, may be partial)
    "box_office_revenue", # 4 – Box-office revenue in USD
    "runtime",            # 5 – Runtime in minutes
    "languages",          # 6 – Freebase JSON {id: language name, ...}
    "countries",          # 7 – Freebase JSON {id: country name, ...}
    "genres",             # 8 – Freebase JSON {id: genre name, ...}
]

CHAR_METADATA_COLS = [
    "wikipedia_id",           # 0 – Wikipedia movie ID (join key)
    "freebase_id",            # 1 – Freebase movie ID
    "release_date",           # 2 – Movie release date
    "character_name",         # 3 – Character name in film
    "actor_dob",              # 4 – Actor date of birth
    "actor_gender",           # 5 – Actor gender
    "actor_height",           # 6 – Actor height in metres
    "actor_ethnicity",        # 7 – Freebase ethnicity ID
    "actor_name",             # 8 – Actor name
    "actor_age_at_release",   # 9 – Actor age at film release
    "freebase_char_actor_id", # 10 – Freebase character/actor map ID
    "freebase_char_id",       # 11 – Freebase character ID
    "freebase_actor_id",      # 12 – Freebase actor ID
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _separator(title: str = "") -> None:
    """Print a section separator to stdout."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    else:
        print(f"{'=' * 60}")


def _log_state(df: pd.DataFrame, step: str, target_col: str = "genre") -> None:
    """Print the dataframe shape and target value counts after a cleaning step.

    Parameters
    ----------
    df:
        Dataframe to inspect.
    step:
        Human-readable description of the cleaning step just completed.
    target_col:
        Column whose value counts to display.  Pass an empty string to skip
        value-count output.
    """
    print(f"\n  [After] {step}")
    print(f"    Shape : {df.shape}")
    if target_col and target_col in df.columns:
        counts = df[target_col].value_counts()
        top = counts.head(10)
        print(f"    '{target_col}' value counts (top {len(top)}):")
        for genre, n in top.items():
            print(f"      {genre:<35} {n:>6}")
        if len(counts) > 10:
            print(f"      ... ({len(counts) - 10} more genres)")


def parse_freebase_json(raw: object) -> list:
    """Convert a Freebase-style JSON string to a list of human-readable values.

    The corpus encodes mappings as Python-literal dicts, e.g.
    ``{"/m/02h40lc": "English Language", "/m/05h43": "Drama"}``.

    Returns an empty list for null, empty, or malformed entries.
    """
    if not isinstance(raw, str) or raw.strip() in ("", "{}"):
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return list(parsed.values())
    except (ValueError, SyntaxError):
        pass
    return []


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------


def download_data() -> None:
    """Download and extract the CMU Movie Summary Corpus if not already present."""
    _separator("Data Acquisition")
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(TARBALL):
        print(f"  Downloading corpus from:\n    {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, TARBALL)
        print(f"  Saved to: {TARBALL}")
    else:
        print(f"  Tarball already present — skipping download.\n    {TARBALL}")

    if not os.path.exists(EXTRACTED):
        print(f"  Extracting {os.path.basename(TARBALL)} …")
        with tarfile.open(TARBALL, "r:gz") as tar:
            tar.extractall(DATA_DIR)
        print(f"  Extracted to: {EXTRACTED}")
    else:
        print(f"  Already extracted — skipping.\n    {EXTRACTED}")


# ---------------------------------------------------------------------------
# Movie metadata cleaning
# ---------------------------------------------------------------------------


def load_movie_metadata() -> pd.DataFrame:
    """Load movie.metadata.tsv and return a cleaned (wikipedia_id, genre) frame.

    Cleaning steps (each logged):
    1. Assign column names; show raw shape.
    2. Parse Freebase JSON genres column into a list of genre strings.
    3. Extract the primary (first) genre per movie as a single label.
    4. Drop rows where no valid genre exists.
    5. Normalise genre strings: strip whitespace, title-case.
    """
    path = os.path.join(EXTRACTED, "movie.metadata.tsv")
    _separator("Loading movie.metadata.tsv")
    print(f"  File: {path}")

    df = pd.read_csv(path, sep="\t", header=None, names=MOVIE_METADATA_COLS)

    print(f"\n  [Before cleaning]")
    print(f"    Shape : {df.shape}")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Null counts:\n{df.isnull().sum().to_string()}")
    print(f"\n  Sample raw 'genres' values:")
    for v in df["genres"].dropna().head(3):
        print(f"    {v!r}")

    # ------------------------------------------------------------------
    # Step 1 – Parse genres column from Freebase JSON
    # ------------------------------------------------------------------
    print("\n  --- Step 1: Parse genres from Freebase JSON ---")
    before = len(df)
    df["genres_list"] = df["genres"].apply(parse_freebase_json)
    n_empty = (df["genres_list"].apply(len) == 0).sum()
    print(f"    Parsed genres for {before} movies.")
    print(f"    Movies with empty genre list: {n_empty}")
    _log_state(df, "Genre list parsed", target_col="")

    # ------------------------------------------------------------------
    # Step 2 – Extract primary genre (first listed)
    # ------------------------------------------------------------------
    print("\n  --- Step 2: Extract primary genre (first entry in list) ---")
    df["genre"] = df["genres_list"].apply(lambda lst: lst[0] if lst else None)
    n_null = df["genre"].isna().sum()
    print(f"    Movies with no primary genre: {n_null}")
    _log_state(df, "Primary genre extracted", target_col="genre")

    # ------------------------------------------------------------------
    # Step 3 – Drop rows with no valid genre
    # ------------------------------------------------------------------
    print("\n  --- Step 3: Drop rows with null or blank genre ---")
    before = len(df)
    df = df[df["genre"].notna() & (df["genre"].str.strip() != "")].copy()
    dropped = before - len(df)
    print(f"    Dropped {dropped} rows.  Remaining: {len(df)}")
    _log_state(df, "No-genre rows removed", target_col="genre")

    # ------------------------------------------------------------------
    # Step 4 – Normalise genre strings
    # ------------------------------------------------------------------
    print("\n  --- Step 4: Normalise genre strings (strip, title-case) ---")
    df["genre"] = df["genre"].str.strip().str.title()
    _log_state(df, "Genre strings normalised", target_col="genre")

    return df[["wikipedia_id", "genre"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Character metadata cleaning
# ---------------------------------------------------------------------------


def load_character_metadata() -> pd.DataFrame:
    """Load character.metadata.tsv and return a (wikipedia_id, character_text) frame.

    character_text is a single space-separated string of all character names
    belonging to that movie — the bag-of-words input feature.

    Cleaning steps (each logged):
    1. Assign column names; show raw shape.
    2. Select wikipedia_id and character_name columns only.
    3. Drop rows with null character names.
    4. Normalise character names: lowercase, remove non-alpha characters, strip.
    5. Drop rows whose names are empty after normalisation.
    6. Group all character names per movie into one string.
    """
    path = os.path.join(EXTRACTED, "character.metadata.tsv")
    _separator("Loading character.metadata.tsv")
    print(f"  File: {path}")

    df = pd.read_csv(path, sep="\t", header=None, names=CHAR_METADATA_COLS)

    print(f"\n  [Before cleaning]")
    print(f"    Shape : {df.shape}")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Null counts in key columns:")
    print(f"      wikipedia_id   : {df['wikipedia_id'].isna().sum()}")
    print(f"      character_name : {df['character_name'].isna().sum()}")
    print(f"\n  Sample raw character names:")
    for v in df["character_name"].dropna().head(5):
        print(f"    {v!r}")

    # ------------------------------------------------------------------
    # Step 1 – Keep only the two columns we need
    # ------------------------------------------------------------------
    print("\n  --- Step 1: Retain wikipedia_id and character_name only ---")
    df = df[["wikipedia_id", "character_name"]].copy()
    _log_state(df, "Column subset selected", target_col="")

    # ------------------------------------------------------------------
    # Step 2 – Drop null character names
    # ------------------------------------------------------------------
    print("\n  --- Step 2: Drop null character names ---")
    before = len(df)
    df = df[df["character_name"].notna()].copy()
    print(f"    Dropped {before - len(df)} null rows.  Remaining: {len(df)}")
    _log_state(df, "Null character names dropped", target_col="")

    # ------------------------------------------------------------------
    # Step 3 – Normalise character names
    # ------------------------------------------------------------------
    print("\n  --- Step 3: Normalise character names ---")
    print("    Operations: cast to str → lowercase → strip → remove non-alpha")
    before = len(df)
    df["character_name"] = (
        df["character_name"]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"[^a-z\s]", "", regex=True)
        .str.strip()
    )
    print(f"\n  Sample normalised character names:")
    for v in df["character_name"].head(5):
        print(f"    {v!r}")

    # ------------------------------------------------------------------
    # Step 4 – Drop names that are blank after normalisation
    # ------------------------------------------------------------------
    print("\n  --- Step 4: Drop empty-after-normalisation character names ---")
    df = df[df["character_name"].str.strip() != ""].copy()
    dropped = before - len(df)
    print(f"    Dropped {dropped} rows now empty.  Remaining: {len(df)}")
    _log_state(df, "Empty names removed", target_col="")

    # ------------------------------------------------------------------
    # Step 5 – Group character names per movie
    # ------------------------------------------------------------------
    print("\n  --- Step 5: Group character names by movie (wikipedia_id) ---")
    df = (
        df.groupby("wikipedia_id")["character_name"]
        .apply(lambda names: " ".join(names))
        .reset_index()
        .rename(columns={"character_name": "character_text"})
    )
    print(f"    Shape after grouping: {df.shape}")
    print(f"\n  Sample character_text strings (first 3 movies):")
    for _, row in df.head(3).iterrows():
        preview = row["character_text"][:80]
        print(f"    wikipedia_id={row['wikipedia_id']}: {preview!r}{'...' if len(row['character_text']) > 80 else ''}")

    return df


# ---------------------------------------------------------------------------
# Feature engineering & classification
# ---------------------------------------------------------------------------


def run_pipeline(df: pd.DataFrame) -> None:
    """Vectorise, train, and evaluate using three vectoriser strategies.

    Parameters
    ----------
    df:
        Merged dataframe with columns: character_text (str), genre (str).
    """
    _separator("Feature Engineering & Classification")

    X = df["character_text"]
    y = df["genre"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    print(f"  Train samples : {len(X_train)}")
    print(f"  Test  samples : {len(X_test)}")
    print(f"  Unique genres : {y.nunique()}")

    vectorisers = {
        "CountVectorizer  (count) ": CountVectorizer(),
        "CountVectorizer  (binary)": CountVectorizer(binary=True),
        "TfidfVectorizer           ": TfidfVectorizer(),
    }

    results: dict[str, tuple[float, LogisticRegression, object]] = {}

    for name, vec in vectorisers.items():
        print(f"\n  --- {name.strip()} ---")
        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)
        print(f"    Vocabulary size : {len(vec.vocabulary_)}")
        print(f"    Feature matrix  : {X_train_vec.shape}")

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_vec, y_train)

        acc = clf.score(X_test_vec, y_test)
        print(f"    Accuracy        : {acc:.4f}  ({acc * 100:.2f}%)")
        results[name] = (acc, clf, vec)

    # ------------------------------------------------------------------
    # Best variant — full classification report
    # ------------------------------------------------------------------
    best_name = max(results, key=lambda k: results[k][0])
    best_acc, best_clf, best_vec = results[best_name]

    _separator(f"Best Variant: {best_name.strip()}")
    print(f"  Accuracy: {best_acc:.4f}  ({best_acc * 100:.2f}%)")

    X_test_best = best_vec.transform(X_test)
    y_pred = best_clf.predict(X_test_best)
    report = classification_report(y_test, y_pred, zero_division=0)
    print("\n  Classification Report:")
    for line in report.splitlines():
        print(f"    {line}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the end-to-end movie genre classification pipeline."""
    _separator("Movie Genre Classification — CMU Movie Summary Corpus")

    # 1. Acquire data
    download_data()

    # 2. Load and clean each file
    movies = load_movie_metadata()
    chars = load_character_metadata()

    # 3. Merge on wikipedia_id
    _separator("Merging Movie Metadata with Character Data")
    print(f"  Movies (after genre cleaning) : {movies.shape}")
    print(f"  Characters (grouped by movie) : {chars.shape}")
    df = movies.merge(chars, on="wikipedia_id", how="inner")
    print(f"  Shape after inner merge       : {df.shape}")

    # Drop any rows where character_text is blank post-merge
    print("\n  Dropping movies with blank character_text after merge …")
    before = len(df)
    df = df[df["character_text"].str.strip() != ""].copy()
    print(f"    Dropped {before - len(df)}.  Final shape: {df.shape}")

    # Final target distribution
    _separator("Final Genre Distribution")
    final_counts = df["genre"].value_counts()
    print(f"  Total movies: {len(df)}")
    print(f"  Unique genres: {df['genre'].nunique()}")
    print(f"\n  All genre counts:")
    for genre, n in final_counts.items():
        bar = "#" * min(40, int(40 * n / final_counts.iloc[0]))
        print(f"    {genre:<35} {n:>5}  {bar}")

    # 4. Feature engineering + classification
    run_pipeline(df)

    _separator("Pipeline Complete")


if __name__ == "__main__":
    main()
