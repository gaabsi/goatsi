from pathlib import Path

import click
import cloudpickle
import joblib
import numpy as np
import pandas as pd
import plotext as plt
import xgboost
from rich.table import Table
from scipy.stats import randint, uniform
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm

from goatsi.src.utils import console, detect_sep, load_defaults, show_centered


class Modelisation:
    """
    Modélisation XGBoost avec hyperparameter search.

    Parametres :
    - train_path (Path) : chemin vers le train set.
    - target (str) : nom de la colonne cible.
    - positive_class (str | None) : valeur de la classe positive si target catégorielle binaire.
    - model (str) : modèle à utiliser, default "xgboost".
    """

    def __init__(
        self,
        train_path: Path,
        target: str,
        positive_class: str | None = None,
        model: str = "xgboost",
    ):
        self.target = target
        self.defaults = load_defaults(model)
        self._models_dir = train_path.parent / "models"

        df = self._load(train_path)
        self.y = df[target]
        self.task = self._infer_task()
        self._encode_target(positive_class)
        self.x = df.drop(columns=target)

    def _load(self, train_path: Path) -> pd.DataFrame:
        """
        Charge le train set depuis le fichier source.

        Parametres :
        - train_path (Path) : chemin vers le fichier.

        Output :
        - (pd.DataFrame) : train set chargé.
        """

        ext = train_path.suffix.lstrip(".")
        readers = {
            "csv": lambda p: pd.read_csv(p, sep=detect_sep(p)),
            "parquet": pd.read_parquet,
            "xlsx": pd.read_excel,
        }

        return readers[ext](train_path)

    def _infer_task(self) -> str:
        """
        Infère la tâche depuis la target : classification si 2 valeurs uniques, sinon régression.

        Output :
        - (str) : "classification" ou "regression".
        """

        return "classification" if self.y.nunique() == 2 else "regression"

    def _encode_target(self, positive_class: str | None) -> None:
        """
        Encode la target en 0/1 si catégorielle binaire.
        Lève une erreur si positive_class est manquante dans ce cas.

        Parametres :
        - positive_class (str | None) : valeur de la classe positive.
        """

        if self.y.dtype == object and self.task == "classification":
            if positive_class is None:
                raise click.UsageError(
                    f"Categorical target detected ({self.y.unique().tolist()}) "
                    f"Specify the --positive-class (ex: '{self.y.unique()[0]}')."
                )
            self.y = self.y.map(lambda v: 1 if v == positive_class else 0)

    def _transtype(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convertit les colonnes object en category.
        """

        obj_cols = df.select_dtypes(include="object").columns.tolist()
        df[obj_cols] = df[obj_cols].astype("category")

        return df

    def _init_pipeline(self) -> Pipeline:
        """
        Construit le pipeline sklearn.
        """

        task_map = {
            "classification": xgboost.XGBClassifier,
            "regression": xgboost.XGBRegressor,
        }
        model = task_map[self.task](
            random_state=self.defaults["seed"],
            enable_categorical=self.defaults["enable_categorical"],
            tree_method=self.defaults["tree_method"],
            n_jobs=self.defaults["threads"],
            objective=(
                "reg:squarederror" if self.task == "regression" else "binary:logistic"
            ),
        )
        return Pipeline(
            [
                ("preprocessor", FunctionTransformer(self._transtype)),
                ("model", model),
            ]
        )

    def _build_param_grid(self) -> dict:
        """
        Convertit les bornes [low, high] du JSON en distributions scipy.
        int -> randint(low, high), float -> uniform(low, high - low).

        Output :
        - (dict) : param grid pour RandomizedSearchCV.
        """

        param_grid = {}
        for param, bounds in self.defaults["param_grid"].items():
            low, high = bounds[0], bounds[1]
            if isinstance(low, int) and isinstance(high, int):
                param_grid[f"model__{param}"] = randint(low, high)
            else:
                param_grid[f"model__{param}"] = uniform(low, high - low)

        return param_grid

    def _fit_best_params(self) -> Pipeline:
        """
        Fit le RandomizedSearchCV et retourne le meilleur pipeline.

        Output :
        - (Pipeline) : pipeline avec les meilleurs hyperparamètres.
        """

        cv_class = KFold if self.task == "regression" else StratifiedKFold
        cv = cv_class(
            n_splits=self.defaults["cv_n_splits"],
            shuffle=True,
            random_state=self.defaults["seed"],
        )
        scoring = {metric: metric for metric in self.defaults["log"]}
        search = RandomizedSearchCV(
            self._init_pipeline(),
            self._build_param_grid(),
            n_iter=self.defaults["n_iter"],
            cv=cv,
            scoring=scoring,
            refit=self.defaults["optimize_on"],
            random_state=self.defaults["seed"],
            n_jobs=-1,
            error_score="raise",
        )

        n_iter = self.defaults["n_iter"]
        n_splits = self.defaults["cv_n_splits"]
        counter = [0]
        bar = tqdm(total=n_iter, desc="Hyperparameter search")

        original = joblib.parallel.BatchCompletionCallBack

        class _Callback(original):
            def __call__(self, *args, **kwargs):
                counter[0] += self.batch_size
                bar.n = counter[0] // n_splits
                bar.refresh()

                return super().__call__(*args, **kwargs)

        joblib.parallel.BatchCompletionCallBack = _Callback
        try:
            search.fit(self.x, self.y)
        finally:
            joblib.parallel.BatchCompletionCallBack = original
            bar.close()

        self._fitted_search = search

        return search.best_estimator_

    def _learning_curve(self, model: Pipeline) -> None:
        """
        Affiche la learning curve du meilleur modèle via plotext.
        """

        metric = self.defaults["optimize_on"]
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model,
            self.x,
            self.y,
            cv=5,
            scoring=metric,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
        )

        train_sizes = train_sizes_abs / len(self.x)
        train_mean = train_scores.mean(axis=1)
        val_mean = val_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_std = val_scores.std(axis=1)

        plt.clf()
        plt.theme("clear")
        plt.plotsize(80, 20)
        plt.plot(
            train_sizes, train_mean, color="blue+", label="Train", marker="braille"
        )
        plt.plot(
            train_sizes, val_mean, color="orange", label="Validation", marker="braille"
        )

        for i, s in enumerate(train_sizes):
            for mean, std, color in [
                (train_mean, train_std, "blue+"),
                (val_mean, val_std, "orange"),
            ]:
                lo, hi = mean[i] - std[i], mean[i] + std[i]
                tige = np.linspace(lo, hi, 10)
                plt.scatter([s] * len(tige), tige, color=color, marker="|")
                plt.scatter([s, s], [lo, hi], color=color, marker="-")

        plt.title(f"Learning Curve — {metric.upper()}")
        plt.xlabel("Taille du train set")
        show_centered(plot_width=80)

    def _save_model(self, model: Pipeline) -> Path:
        """
        Sauvegarde le modèle dans models/model_<index>.pkl.

        Output :
        - (Path) : chemin du modèle sauvegardé.
        """

        self._models_dir.mkdir(exist_ok=True)
        index = len(list(self._models_dir.glob("model_*.pkl")))
        model_path = self._models_dir / f"model_{index}.pkl"
        with open(model_path, "wb") as f:
            cloudpickle.dump(model, f)
        console.print(
            f"[green]✓ Modèle sauvegardé → {model_path}[/green]", justify="center"
        )

        return model_path

    def run(self) -> Path:
        """
        Orchestration : fit, log des métriques CV, learning curve, sauvegarde.

        Output :
        - (Path) : chemin du modèle sauvegardé.
        """

        best_model = self._fit_best_params()
        search = self._fitted_search

        table = Table()
        table.add_column("Métrique", style="cyan", justify="center")
        table.add_column("Score moyen (CV)", style="green", justify="center")
        for metric in self.defaults["log"]:
            mean_score = search.cv_results_[f"mean_test_{metric}"][search.best_index_]
            table.add_row(metric, f"{mean_score:.4f}")
        console.print(table, justify="center")

        self._learning_curve(best_model)

        return self._save_model(best_model)
