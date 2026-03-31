import csv
import importlib.resources
import json
import shutil
from pathlib import Path

import plotext as plt
from rich.console import Console

console = Console()


def load_defaults(model: str = "xgboost") -> dict:
    """
    Charge les paramètres par défaut depuis src/defaults.json.

    Parametres :
    - model (str) : nom du modèle dont on veut les defaults.

    Output :
    - (dict) : paramètres par défaut du modèle.
    """
    with importlib.resources.open_text("goatsi.src", "defaults.json") as f:
        return json.load(f)[model]


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
