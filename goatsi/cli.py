import ast
from pathlib import Path

import click

from goatsi.commands.split import Ingestion


@click.group()
def cli():
    pass


@cli.command()
@click.argument("filepath", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-target",
    default=None,
    help="Colonne cible. Active la stratification automatiquement.",
)
@click.option(
    "-train-size", default=0.8, show_default=True, help="Proportion du train set."
)
@click.option(
    "-usecols", default=None, help="Colonnes à garder, ex: \"['col1', 'col2']\"."
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
                "Format attendu : \"['col1', 'col2']\"", param_hint="--usecols"
            )

    Ingestion(
        filepath=filepath,
        target=target,
        train_size=train_size,
        stratified=target is not None,
        usecols=usecols,
    ).run()
