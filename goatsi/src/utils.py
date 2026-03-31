import csv
from pathlib import Path

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
