"""Tests for nextflow/bin/dispatch_cli.py.

dispatch_cli.py is a script, not a package member, and its COMMANDS dict
points at modules that import heavy GPU deps (cupy, torch, cellpose, viscy).
We load it via importlib.util to avoid resolving any of those imports, and we
only inspect the COMMANDS table + the _type_from_annotation helper.
"""

import importlib.util
import re
import sys
import typing
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DISPATCH_PATH = REPO_ROOT / "nextflow" / "bin" / "dispatch_cli.py"


@pytest.fixture(scope="module")
def dispatch_cli():
    spec = importlib.util.spec_from_file_location("dispatch_cli", DISPATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass annotation resolution can find the
    # module namespace via sys.modules[cls.__module__] (Python 3.12).
    sys.modules["dispatch_cli"] = module
    spec.loader.exec_module(module)
    return module


class TestFunctionsTable:
    def test_keys_are_unique(self, dispatch_cli):
        keys = list(dispatch_cli.COMMANDS.keys())
        assert len(keys) == len(set(keys))

    def test_keys_are_valid_argparse_choices(self, dispatch_cli):
        # argparse choices are matched as strings; ensure no whitespace or
        # shell-unfriendly chars that would break Nextflow invocation.
        bad = [
            k for k in dispatch_cli.COMMANDS if not re.fullmatch(r"[A-Za-z0-9_\-]+", k)
        ]
        assert bad == [], f"command names with invalid chars: {bad}"

    def test_values_are_module_func_pairs(self, dispatch_cli):
        module_re = re.compile(r"^cyclops_process(\.[A-Za-z_][A-Za-z0-9_]*)+$")
        func_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        for cmd, value in dispatch_cli.COMMANDS.items():
            module_path, func_name = value.module, value.func
            assert module_re.fullmatch(
                module_path
            ), f"{cmd}: bad module path {module_path!r}"
            assert func_re.fullmatch(func_name), f"{cmd}: bad func name {func_name!r}"

    def test_nonempty(self, dispatch_cli):
        assert len(dispatch_cli.COMMANDS) > 0


class TestTypeFromAnnotation:
    def test_empty_annotation_defaults_to_str(self, dispatch_cli):
        import inspect

        assert dispatch_cli._type_from_annotation(inspect.Parameter.empty) is str

    def test_plain_int(self, dispatch_cli):
        assert dispatch_cli._type_from_annotation(int) is int

    def test_plain_float(self, dispatch_cli):
        assert dispatch_cli._type_from_annotation(float) is float

    def test_plain_str(self, dispatch_cli):
        assert dispatch_cli._type_from_annotation(str) is str

    def test_optional_int_unwraps(self, dispatch_cli):
        assert dispatch_cli._type_from_annotation(typing.Optional[int]) is int

    def test_optional_str_unwraps(self, dispatch_cli):
        assert dispatch_cli._type_from_annotation(typing.Optional[str]) is str

    def test_union_of_many_falls_back_to_str(self, dispatch_cli):
        # Union with >1 non-None member can't be coerced — must fall back.
        assert dispatch_cli._type_from_annotation(typing.Union[int, float, str]) is str

    def test_literal_falls_back_to_str(self, dispatch_cli):
        assert (
            dispatch_cli._type_from_annotation(typing.Literal["track", "pheno"]) is str
        )

    def test_unknown_type_falls_back_to_str(self, dispatch_cli):
        class Custom:
            pass

        assert dispatch_cli._type_from_annotation(Custom) is str

    @pytest.mark.parametrize(
        "falsy", ["false", "False", "FALSE", "0", "no", "No", "NO"]
    )
    def test_bool_falsy_strings(self, dispatch_cli, falsy):
        fn = dispatch_cli._type_from_annotation(bool)
        assert fn(falsy) is False

    @pytest.mark.parametrize("truthy", ["true", "True", "1", "yes", "anything-else"])
    def test_bool_truthy_strings(self, dispatch_cli, truthy):
        fn = dispatch_cli._type_from_annotation(bool)
        assert fn(truthy) is True
