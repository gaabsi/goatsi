from pathlib import Path

import pandas as pd
from rich.table import Table
from sklearn.model_selection import train_test_split

from goatsi.src.utils import console, detect_sep


class Ingestion:
    """
    Ingestion du dataset et split train/test.

    Parametres :
    - filepath (Path) : chemin vers le fichier de données.
    - target (str) : nom de la colonne cible.
    - train_size (float) : proportion du train set.
    - seed (int) : graine aléatoire.
    - stratified (bool) : si True, stratifie le split sur la target.
    - usecols (list[str] | None) : colonnes à charger, None = toutes.
    """

    def __init__(
        self,
        filepath: Path,
        target: str,
        train_size: float = 0.8,
        seed: int = 77,
        stratified: bool = False,
        usecols: list[str] | None = None,
    ):
        self.filepath = filepath
        self.target = target
        self.train_size = train_size
        self.seed = seed
        self.stratified = stratified
        self.usecols = usecols
        self.extension = filepath.suffix.lstrip(".")

    @staticmethod
    def _load(
        filepath: Path,
        sep: str,
        usecols: list[str] | None = None,
        n_rows: int | None = None,
    ) -> pd.DataFrame:
        """
        Charge le dataset depuis le fichier source.

        Parametres :
        - filepath (Path) : chemin vers le fichier.
        - sep (str) : séparateur CSV (ignoré pour parquet/xlsx).
        - usecols (list[str] | None) : colonnes à charger, None = toutes.
        - n_rows (int | None) : nombre de lignes à charger, None = toutes.

        Output :
        - (pd.DataFrame) : dataset chargé.
        """

        ext = filepath.suffix.lstrip(".")
        readers = {
            "csv": lambda p: pd.read_csv(p, sep=sep, usecols=usecols, nrows=n_rows),
            "parquet": lambda p: (
                pd.read_parquet(p, columns=usecols).head(n_rows)
                if n_rows
                else pd.read_parquet(p, columns=usecols)
            ),
            "xlsx": lambda p: pd.read_excel(p, usecols=usecols, nrows=n_rows),
        }
        return readers[ext](filepath)

    def split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Charge le dataset et retourne les splits train et test.

        Output :
        - (train_set, test_set) (tuple) : datasets d'entraînement et de test.
        """

        sep = detect_sep(self.filepath) if self.extension == "csv" else ","
        df = Ingestion._load(self.filepath, sep, usecols=self.usecols)
        x = df.drop(columns=[self.target])
        y = df[self.target]

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            train_size=self.train_size,
            random_state=self.seed,
            stratify=y if self.stratified else None,
        )

        return pd.concat([x_train, y_train], axis=1), pd.concat(
            [x_test, y_test], axis=1
        )

    def _log_split(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """
        Affiche les caractéristiques du split dans une table Rich.
        """

        table = Table()
        table.add_column("train size", style="cyan")
        table.add_column("stratified", style="cyan")
        table.add_column("train shape", style="cyan")
        table.add_column("test shape", style="cyan")
        table.add_row(
            str(self.train_size),
            str(self.stratified),
            str(train_set.shape),
            str(test_set.shape),
        )
        console.print(table, justify="center")

    def write(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """
        Écrit les datasets train et test dans le même dossier que la source.
        Même extension que le fichier source.

        Parametres :
        - train_set (pd.DataFrame) : dataset d'entraînement.
        - test_set (pd.DataFrame) : dataset de test.
        """

        writers = {
            "csv": lambda df, p: df.to_csv(
                p, index=False, sep=detect_sep(self.filepath)
            ),
            "parquet": lambda df, p: df.to_parquet(p, index=False),
            "xlsx": lambda df, p: df.to_excel(p, index=False),
        }
        base_path = self.filepath.parent
        writer = writers[self.extension]
        writer(train_set, base_path / f"train_set.{self.extension}")
        writer(test_set, base_path / f"test_set.{self.extension}")
        console.print(
            f"[green]✓ Datasets sauvegardés dans {base_path}[/green]", justify="center"
        )

    def run(self) -> None:
        """
        Orchestre le chargement, le split et l'écriture des datasets.
        """

        train_set, test_set = self.split()
        self._log_split(train_set, test_set)
        self.write(train_set, test_set)
