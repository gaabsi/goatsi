import csv
import json
import shutil
from pathlib import Path

import click
import cloudpickle
import pandas as pd
import plotext as plt
from rich.console import Console

console = Console()


def detect_sep(filepath: Path) -> str:
    """
    Détecte le séparateur d'un fichier CSV via csv.Sniffer.

    Parametres :
    - filepath (Path) : chemin vers le fichier CSV.

    Output :
    - (str) : séparateur détecté.
    """
    with open(filepath, newline="", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except csv.Error:
        raise ValueError(
            f"Impossible de détecter le séparateur de {filepath.name}. Vérifier le fichier."
        )


def load_defaults(model: str = "xgboost") -> dict:
    """
    Charge les paramètres par défaut depuis src/defaults.json.

    Parametres :
    - model (str) : nom du modèle dont on veut les defaults.

    Output :
    - (dict) : paramètres par défaut du modèle.
    """
    with open(Path(__file__).parent / "defaults.json") as f:
        data = json.load(f)
    return {**data["tasks"], **data["models"][model]}


def show_centered(plot_width: int = 70) -> None:
    """
    Affiche un plot plotext centré dans le terminal.

    Parametres :
    - plot_width (int) : largeur du plot en caractères.
    """
    padding = " " * max(0, (shutil.get_terminal_size().columns - plot_width) // 2)
    for line in plt.build().split("\n"):
        print(padding + line)
    plt.clf()


def encode_target(y: pd.Series, positive_class: str | None) -> pd.Series:
    """
    Encode la target en 0/1 si elle est de type object (catégorielle binaire).
    Si la target est déjà numérique, retourne y sans modification.

    Parametres :
    - y (pd.Series) : colonne cible.
    - positive_class (str | None) : valeur de la classe positive.

    Output :
    - (pd.Series) : target encodée.
    """

    if y.dtype != object:
        return y

    if positive_class is None:
        raise click.UsageError(
            f"Categorical target detected ({y.unique().tolist()}). "
            f"Specify --positive-class (ex: '{y.unique()[0]}')."
        )

    return y.map(lambda v: 1 if v == positive_class else 0)


def load_model(model_path: Path):
    """
    Charge un modèle depuis un fichier .pkl avec cloudpickle.

    Parametres :
    - model_path (Path) : chemin vers le fichier .pkl.

    Output :
    - pipeline chargé.
    """
    with open(model_path, "rb") as f:
        return cloudpickle.load(f)


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Charge un dataset depuis un fichier csv, parquet ou xlsx.

    Parametres :
    - path (Path) : chemin vers le fichier.

    Output :
    - (pd.DataFrame) : dataset chargé.
    """
    ext = path.suffix.lstrip(".")
    readers = {
        "csv": lambda p: pd.read_csv(p, sep=detect_sep(p)),
        "parquet": pd.read_parquet,
        "xlsx": pd.read_excel,
    }

    return readers[ext](path)
