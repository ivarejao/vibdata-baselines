from pathlib import Path

import typer

from .sklearn import main as _main_sklearn
from .baselines import main as _main_baselines

_app = typer.Typer(pretty_exceptions_show_locals=False)


@_app.command(name="baselines")
def _main_baselines_wrapper(
    cfg: Path = typer.Option(help="Config file"),
    biased: bool = typer.Option(False, help="Use biased classifier"),
):
    _main_baselines(cfg, biased)


@_app.command(name="experiment")
def _main_sklearn_wrapper(
    cfg: Path = typer.Option(help="Config file"),
    biased: bool = typer.Option(False, help="Use biased classifier")
):
    _main_sklearn(cfg, biased)


def run_baselines():
    _app()
