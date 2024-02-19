import json
from os import PathLike
from typing import Any, Literal
from importlib.resources import path as resource_path

import yaml
from jsonschema import validate
from jsonschema.exceptions import ValidationError


def load_config(config_path: str | PathLike) -> tuple[Literal["deep", "classic"], dict[str, Any]]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    current_module = __spec__.parent
    with resource_path(current_module, "classic.schema.json") as schema_path:
        with open(schema_path, "r") as f:
            classic_schema = json.load(f)

    with resource_path(current_module, "deep.schema.json") as schema_path:
        with open(schema_path, "r") as f:
            deep_schema = json.load(f)

    error_classic = None
    error_deep = None

    try:
        validate(config, schema=deep_schema)
    except ValidationError as e:
        error_deep = e

    try:
        validate(config, schema=classic_schema)
    except ValidationError as e:
        error_classic = e

    # This statement exists for debug porpuses
    # This exception should never be raised
    if error_classic is None and error_deep is None:
        raise RuntimeError("Both schemas are valid")

    if error_classic is None:
        return "classic", config
    elif error_deep is None:
        return "deep", config

    raise Exception([error_classic, error_deep])
