"""Central storage roots for cyclops_process.

All pipeline outputs live under ``BASE_PATH``, which must be supplied via the
``OPS_BASE_PATH`` environment variable — there is no default, so the package
never silently writes to somebody else's storage:

    export OPS_BASE_PATH=/path/to/ops_data

Raw acquisitions are read from a separate instrument mount, supplied via
``OPS_INSTRUMENT_ROOT``. That one is resolved lazily through
:func:`instrument_root` because only the conversion steps need it.
"""
import os


def _require(var: str) -> str:
    """Return env var ``var``, or raise with a usable message if it is unset."""
    value = os.environ.get(var)
    if not value:
        raise RuntimeError(
            f"{var} is not set. Point it at your storage root, e.g. "
            f"`export {var}=/path/to/ops_data`."
        )
    return value


def instrument_root() -> str:
    """Root of the raw instrument acquisitions (``OPS_INSTRUMENT_ROOT``).

    Resolved on call rather than at import so that only the steps which read
    raw data require it to be configured.
    """
    return _require("OPS_INSTRUMENT_ROOT")


BASE_PATH = _require("OPS_BASE_PATH")
