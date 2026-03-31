import ast
from pathlib import Path

import click

from goatsi.commands.eval import Evaluation
from goatsi.commands.fit import Modelisation
from goatsi.commands.split import Ingestion


@click.group()
def cli():
    pass


@cli.command()
@click.argument("filepath", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-target",
    default=None,
    help="Target column. Enables stratification automatically.",
)
@click.option(
    "-train-size", default=0.8, show_default=True, help="Train set proportion."
)
@click.option(
    "-usecols", default=None, help="Columns to keep, ex: \"['col1', 'col2']\"."
)
def split(filepath, target, train_size, usecols):
    """
    Sépare un dataset en jeux train et test.
    """

    if usecols is not None:
        try:
            usecols = ast.literal_eval(usecols)
        except (ValueError, SyntaxError):
            raise click.BadParameter(
                "Expected format: \"['col1', 'col2']\"", param_hint="--usecols"
            )

    Ingestion(
        filepath=filepath,
        target=target,
        train_size=train_size,
        stratified=target is not None,
        usecols=usecols,
    ).run()


@cli.command()
@click.argument("train_path", type=click.Path(exists=True, path_type=Path))
@click.option("--target", "-t", required=True, help="Target column.")
@click.option(
    "--positive-class",
    "-p",
    default=None,
    help="Positive class value (ex: 'Yes').",
)
def fit(train_path, target, positive_class):
    """
    Entraîne un modèle XGBoost sur le train set.
    """

    Modelisation(
        train_path=train_path,
        target=target,
        positive_class=positive_class,
    ).run()


@cli.command()
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.argument("test_path", type=click.Path(exists=True, path_type=Path))
@click.option("--target", "-t", required=True, help="Target column.")
@click.option(
    "--positive-class",
    "-p",
    default=None,
    help="Positive class value (ex: 'Yes'). Required if target is categorical.",
)
def eval(model_path, test_path, target, positive_class):
    """
    Évalue un modèle sur le test set.
    """

    Evaluation(
        model_path=model_path,
        test_path=test_path,
        target=target,
        positive_class=positive_class,
    ).run()
