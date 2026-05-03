"""Preprocessing utilities for the UCI Paddy Dataset project.

The clustering workflow should not use paddy yield as an input feature,
because yield is better kept as an outcome for interpreting clusters.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_TARGET_COLUMN = "Paddy yield(in Kg)"


@dataclass(frozen=True)
class PreprocessingResult:
    """Container for transformed features and the fitted preprocessing object."""

    features: pd.DataFrame
    target: pd.Series | None
    metadata: dict
    transformer: ColumnTransformer


def project_root_from_file() -> Path:
    """Return the project root assuming this file lives in src/."""

    return Path(__file__).resolve().parents[1]


def load_paddy_dataset(path: str | Path) -> pd.DataFrame:
    """Load the raw Paddy Dataset and normalize column names."""

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def feature_columns(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    drop_columns: Iterable[str] | None = None,
) -> list[str]:
    """Return modeling feature columns after excluding target/drop columns."""

    excluded = set(drop_columns or [])
    excluded.add(target_column)
    return [column for column in df.columns if column not in excluded]


def split_feature_types(df: pd.DataFrame, columns: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split selected columns into numeric and categorical groups."""

    selected = list(columns)
    numeric_cols = df[selected].select_dtypes(include="number").columns.tolist()
    categorical_cols = df[selected].select_dtypes(exclude="number").columns.tolist()
    return numeric_cols, categorical_cols


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense one-hot encoder across scikit-learn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    numeric_columns: Iterable[str],
    categorical_columns: Iterable[str],
) -> ColumnTransformer:
    """Build the preprocessing transformer used before PCA/clustering."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_columns)),
            ("categorical", categorical_pipeline, list(categorical_columns)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def preprocess_paddy_dataset(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    drop_columns: Iterable[str] | None = None,
) -> PreprocessingResult:
    """Fit preprocessing and return a transformed feature matrix.

    The target column is excluded from features by default and returned
    separately for later cluster interpretation.
    """

    columns = feature_columns(df, target_column=target_column, drop_columns=drop_columns)
    numeric_cols, categorical_cols = split_feature_types(df, columns)
    transformer = build_preprocessor(numeric_cols, categorical_cols)
    transformed = transformer.fit_transform(df[columns])
    output_columns = transformer.get_feature_names_out()
    features = pd.DataFrame(transformed, columns=output_columns, index=df.index)
    target = df[target_column].copy() if target_column in df.columns else None
    metadata = {
        "rows": int(df.shape[0]),
        "raw_columns": int(df.shape[1]),
        "feature_columns_before_encoding": len(columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "encoded_feature_columns": int(features.shape[1]),
        "target_column": target_column if target_column in df.columns else None,
        "excluded_columns": sorted(set(df.columns) - set(columns)),
        "missing_values_in_raw_data": int(df.isna().sum().sum()),
        "duplicate_rows_in_raw_data": int(df.duplicated().sum()),
    }
    return PreprocessingResult(
        features=features,
        target=target,
        metadata=metadata,
        transformer=transformer,
    )


def save_preprocessing_outputs(
    result: PreprocessingResult,
    output_dir: str | Path,
    table_dir: str | Path,
) -> None:
    """Save processed features and metadata for downstream notebooks."""

    output_path = Path(output_dir)
    table_path = Path(table_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    table_path.mkdir(parents=True, exist_ok=True)

    result.features.to_csv(output_path / "processed_features.csv", index=False)
    if result.target is not None:
        result.target.to_frame().to_csv(output_path / "target_yield.csv", index=False)

    feature_schema = pd.DataFrame(
        {
            "processed_feature": result.features.columns,
            "feature_order": range(result.features.shape[1]),
        }
    )
    feature_schema.to_csv(table_path / "processed_feature_schema.csv", index=False)
    (table_path / "preprocessing_metadata.json").write_text(
        json.dumps(result.metadata, indent=2),
        encoding="utf-8",
    )


def run_preprocessing(
    data_path: str | Path,
    output_dir: str | Path,
    table_dir: str | Path,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> PreprocessingResult:
    """Load, preprocess, and save the Paddy Dataset."""

    df = load_paddy_dataset(data_path)
    result = preprocess_paddy_dataset(df, target_column=target_column)
    save_preprocessing_outputs(result, output_dir=output_dir, table_dir=table_dir)
    return result


def parse_args() -> argparse.Namespace:
    root = project_root_from_file()
    parser = argparse.ArgumentParser(description="Preprocess the Paddy Dataset.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=root / "data" / "paddydataset.csv",
        help="Path to the raw Paddy Dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "processed",
        help="Directory for processed feature outputs.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=root / "outputs" / "tables",
        help="Directory for metadata and schema tables.",
    )
    parser.add_argument(
        "--target-column",
        default=DEFAULT_TARGET_COLUMN,
        help="Outcome column to exclude from clustering features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_preprocessing(
        data_path=args.data_path,
        output_dir=args.output_dir,
        table_dir=args.table_dir,
        target_column=args.target_column,
    )
    print(f"Processed rows: {result.features.shape[0]:,}")
    print(f"Processed feature columns: {result.features.shape[1]:,}")
    print(f"Target column excluded: {result.metadata['target_column']}")


if __name__ == "__main__":
    main()
