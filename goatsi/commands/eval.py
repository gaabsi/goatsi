from pathlib import Path

import plotext as plt
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from goatsi.src.utils import (
    console,
    encode_target,
    load_dataset,
    load_model,
    show_centered,
)


class Evaluation:
    """
    Évalue les performances d'un modèle de classification binaire sur un test set.

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

        self.y_pred = None
        self.y_pred_prob = None
        self.metrics = None

    def _predict(self) -> None:
        """
        Calcule les prédictions binaires et les probabilités.
        """

        self.y_pred = self.pipeline.predict(self.x_test)
        self.y_pred_prob = self.pipeline.predict_proba(self.x_test)[:, 1]

    def _compute_metrics(self) -> None:
        """
        Calcule les métriques d'évaluation sur le test set.
        """

        self.metrics = {
            "Accuracy": accuracy_score(self.y_test, self.y_pred),
            "Precision": precision_score(self.y_test, self.y_pred),
            "Recall": recall_score(self.y_test, self.y_pred),
            "F1": f1_score(self.y_test, self.y_pred),
            "AUC-ROC": roc_auc_score(self.y_test, self.y_pred_prob),
            "Avg Precision": average_precision_score(self.y_test, self.y_pred_prob),
        }

    def _show_metrics(self) -> None:
        """
        Affiche les métriques sous forme de table rich.
        """

        table = Table(title="Evaluation metrics")
        for key in self.metrics:
            table.add_column(key, style="cyan", justify="center")
        table.add_row(*[f"{v:.3f}" for v in self.metrics.values()])
        console.print(table, justify="center")

    def _show_confusion(self) -> None:
        """
        Affiche la matrice de confusion sous forme de table rich.
        """

        cm = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = cm.ravel()

        table = Table(title="Confusion matrix")
        table.add_column("", style="bold")
        table.add_column("Predicted 0", justify="center")
        table.add_column("Predicted 1", justify="center")
        table.add_row("Actual 0", f"[green]{tn}[/green]", f"[red]{fp}[/red]")
        table.add_row("Actual 1", f"[red]{fn}[/red]", f"[green]{tp}[/green]")
        console.print(table, justify="center")

    def _show_roc(self) -> None:
        """
        Affiche la courbe ROC dans le terminal via plotext.
        """

        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_prob)
        plt.clf()
        plt.theme("clear")
        plt.plotsize(70, 20)
        plt.plot(
            fpr.tolist(),
            tpr.tolist(),
            color="blue+",
            label=f"ROC (AUC: {self.metrics['AUC-ROC']:.3f})",
            marker="braille",
        )
        plt.plot([0, 1], [0, 1], color="red", label="Random", marker="braille")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        show_centered(plot_width=70)

    def _show_pr(self) -> None:
        """
        Affiche la courbe Précision-Rappel dans le terminal via plotext.
        """

        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_prob)
        plt.clf()
        plt.theme("clear")
        plt.plotsize(70, 20)
        plt.plot(
            recall.tolist(),
            precision.tolist(),
            color="orange",
            label=f"P-R (AP: {self.metrics['Avg Precision']:.3f})",
            marker="braille",
        )
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        show_centered(plot_width=70)

    def _show_proba(self) -> None:
        """
        Affiche la distribution des probabilités prédites par classe réelle.
        """

        prob_0 = self.y_pred_prob[self.y_test == 0].tolist()
        prob_1 = self.y_pred_prob[self.y_test == 1].tolist()
        plt.clf()
        plt.theme("clear")
        plt.plotsize(70, 20)
        plt.hist(prob_0, bins=50, color="blue+", label="True label 0")
        plt.hist(prob_1, bins=50, color="orange", label="True label 1")
        plt.title("Predicted probability distribution")
        plt.xlabel("Predicted probability")
        plt.xlim(0, 1)
        show_centered(plot_width=70)

    def run(self) -> None:
        """
        Orchestration : prédictions, métriques, visualisations.
        """

        self._predict()
        self._compute_metrics()
        self._show_metrics()
        self._show_confusion()
        self._show_roc()
        self._show_pr()
        self._show_proba()
