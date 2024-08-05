from pathlib import Path

import typer

from vibnet.data.multi_rounds import main as _main_mr_data_division

from .common import Split
from .sklearn import main as _main_sklearn
from .baselines import main as _main_baselines

_app = typer.Typer(pretty_exceptions_show_locals=False)


@_app.command(name="baselines")
def _main_baselines_wrapper(
    cfg: Path = typer.Option(help="Config file"),
    split: Split = typer.Option(default=Split.unbiased, help="Type of division"),
    clear_cache: bool = typer.Option(default=False, is_flag=True, help="Clear the cache data before running"),
):
    _main_baselines(cfg, split, clear_cache)


@_app.command(name="experiment")
def _main_sklearn_wrapper(
    cfg: Path = typer.Option(help="Config file"),
    split: Split = typer.Option(default=Split.unbiased, help="Type of division"),
    clear_cache: bool = typer.Option(default=False, is_flag=True, help="Clear the cache data before running"),
):
    _main_sklearn(cfg, split, clear_cache)


@_app.command(name="mr_data_division")
def _main_mr_data_division_wrapper(
    cfg: Path = typer.Option(help="Config file"),
    clear_cache: bool = typer.Option(default=False, is_flag=True, help="Clear the cache data before running"),
):
    _main_mr_data_division(cfg, clear_cache)


def run_baselines():
    _app()
