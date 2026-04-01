from pathlib import Path

import numpy as np
import pandas as pd
import shap
from rich.table import Table

from goatsi.src.utils import console, encode_target, load_dataset, load_model


def _make_bar(value: float, max_val: float, width: int = 28) -> str:
    """
    Génère une barre ASCII proportionnelle à la valeur.

    Parametres :
    - value (float) : valeur absolue à représenter.
    - max_val (float) : valeur maximale pour normaliser.
    - width (int) : largeur max de la barre en caractères.

    Output :
    - (str) : barre de caractères ▬.
    """
    n = int(abs(value) / max_val * width) if max_val > 0 else 0
    return "▬" * n


class Explanation:
    """
    Explique les prédictions d'un modèle tree-based via SHAP.

    Parametres :
    - model_path (Path) : chemin vers le fichier .pkl du modèle.
    - test_path (Path) : chemin vers le fichier test set.
    - target (str) : nom de la colonne cible.
    - positive_class (str | None) : valeur de la classe positive si target catégorielle.
    """

    def __init__(
        self,
        model_path: Path,
        test_path: Path,
        target: str,
        positive_class: str | None = None,
    ):
        self.pipeline = load_model(model_path)

        df = load_dataset(test_path)
        y = df[target]
        self.y_test = encode_target(y, positive_class)
        self.x_test = df.drop(columns=target)

        self.task = "classification" if self.y_test.nunique() == 2 else "regression"
        self.shap_values = None
        self.x_processed = None
        self.y_pred = None

    def _check_tree_based(self) -> bool:
        """
        Vérifie si le modèle est tree-based (supporte SHAP TreeExplainer).

        Output :
        - (bool) : True si tree-based, False sinon.
        """
        return hasattr(self.pipeline.named_steps["model"], "feature_importances_")

    def _get_shap_values(self) -> None:
        """
        Applique les transformations du pipeline puis calcule les valeurs SHAP.
        """
        self.x_processed = self.x_test.copy()
        for _, transformer in self.pipeline.steps[:-1]:
            self.x_processed = transformer.transform(self.x_processed)

        model = self.pipeline.named_steps["model"]
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(self.x_processed)

    def _get_regression_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Pour la régression, retourne les indices du bottom et top 25% des prédictions.

        Output :
        - (tuple) : (indices_low, indices_high).
        """
        self.y_pred = self.pipeline.predict(self.x_test)
        q25 = np.percentile(self.y_pred, 25)
        q75 = np.percentile(self.y_pred, 75)
        indices_low = np.where(self.y_pred <= q25)[0]
        indices_high = np.where(self.y_pred >= q75)[0]
        return indices_low, indices_high

    def _shap_summary(self) -> None:
        """
        Table Rich des top 8 features par mean(|shap value|), avec direction d'effet.
        """
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        features = self.x_test.columns.tolist()
        indices = np.argsort(mean_shap)[-8:][::-1]
        top_features = [features[i] for i in indices]
        top_values = mean_shap[indices]
        max_val = top_values.max()

        table = Table(title="SHAP — Top 8 features (mean |shap value|)")
        table.add_column("Feature", style="cyan", justify="left")
        table.add_column("Direction", justify="center")
        table.add_column("Importance", justify="left", min_width=30)
        table.add_column("Score", style="magenta", justify="right")

        for feat, val in zip(top_features, top_values):
            feat_idx = features.index(feat)
            feat_num = pd.to_numeric(self.x_test[feat], errors="coerce").fillna(0)
            n = len(feat_num)
            q = max(1, n // 4)
            sorted_idx = np.argsort(feat_num.values)
            high_shap = self.shap_values[sorted_idx[-q:], feat_idx].mean()
            low_shap = self.shap_values[sorted_idx[:q], feat_idx].mean()
            diff = high_shap - low_shap

            if np.isnan(diff) or abs(diff) < 0.05:
                direction = "[white]?[/white]"
            elif diff > 0:
                direction = "[green]↑[/green]"
            else:
                direction = "[red]↓[/red]"

            bar = _make_bar(val, max_val)
            table.add_row(feat, direction, f"[magenta]{bar}[/magenta]", f"{val:.3f}")

        console.print(table, justify="center")

    def _waterfall(self, title: str, candidates: np.ndarray) -> None:
        """
        Table Rich des contributions SHAP pour une observation tirée aléatoirement parmi les candidats.

        Parametres :
        - title (str) : titre de la table.
        - candidates (np.ndarray) : indices candidats parmi lesquels tirer l'observation.
        """
        idx = np.random.choice(candidates)
        shap_vals = self.shap_values[idx]
        features = self.x_test.columns.tolist()
        order = np.argsort(np.abs(shap_vals))[-8:][::-1]
        f_names = [features[i] for i in order]
        f_vals = shap_vals[order]
        max_val = np.abs(f_vals).max()

        table = Table(title=title)
        table.add_column("Feature", style="cyan", justify="left")
        table.add_column("Value", style="white", justify="right")
        table.add_column("Contribution", justify="left", min_width=30)
        table.add_column("SHAP", justify="right")

        for feat, val in zip(f_names, f_vals):
            color = "green" if val >= 0 else "red"
            bar = _make_bar(val, max_val)
            sign = "+" if val >= 0 else ""
            raw = self.x_test.iloc[idx][feat]
            val_str = (
                f"{raw:.3g}"
                if isinstance(raw, (int, float, np.integer, np.floating))
                else str(raw)
            )
            table.add_row(
                feat,
                val_str,
                f"[{color}]{bar}[/{color}]",
                f"[{color}]{sign}{val:.3f}[/{color}]",
            )

        console.print(table, justify="center")

    def run(self) -> None:
        """
        Orchestration : vérifie le modèle, calcule SHAP, affiche summary et 2 waterfalls.
        """
        if not self._check_tree_based():
            console.print(
                "[red]✗ Ce modèle n'est pas tree-based — SHAP non disponible.[/red]"
            )
            return

        self._get_shap_values()
        self._shap_summary()

        if self.task == "classification":
            self._waterfall(
                title="SHAP Waterfall — negative case (y=0)",
                candidates=np.where(np.array(self.y_test) == 0)[0],
            )
            self._waterfall(
                title="SHAP Waterfall — positive case (y=1)",
                candidates=np.where(np.array(self.y_test) == 1)[0],
            )
        else:
            indices_low, indices_high = self._get_regression_indices()
            self._waterfall(
                title="SHAP Waterfall — low prediction (bottom 25%)",
                candidates=indices_low,
            )
            self._waterfall(
                title="SHAP Waterfall — high prediction (top 25%)",
                candidates=indices_high,
            )
