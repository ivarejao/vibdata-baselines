from pathlib import Path

import typer

from .common import Split
from .sklearn import main as _main_sklearn
from .baselines import main as _main_baselines

_app = typer.Typer(pretty_exceptions_show_locals=False)


@_app.command(name="baselines")
def _main_baselines_wrapper(
    cfg: Path = typer.Option(help="Config file"),
    split: Split = typer.Option(default=Split.unbiased, help="Type of division"),
    clear_cache: bool = typer.Option(help="Clear the cache data before running"),
):
    _main_baselines(cfg, split)


@_app.command(name="experiment")
def _main_sklearn_wrapper(
    cfg: Path = typer.Option(help="Config file"),
    split: Split = typer.Option(default=Split.unbiased, help="Type of division"),
    clear_cache: bool = typer.Option(help="Clear the cache data before running"),
):
    _main_sklearn(cfg, split)


def run_baselines():
    _app()
