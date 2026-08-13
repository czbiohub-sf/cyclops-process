import inspect
import os
import sys
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel

sys.path.insert(0, os.getcwd())

# Lazy load waveorder focus function
_wo_focus = None

# Path to GPU-enabled waveorder with batched focus + device support
_GPU_WAVEORDER_PATH = os.environ.get("OPS_GPU_WAVEORDER_PATH", "")


def _get_wo_focus():
    """Lazy load waveorder focus function, preferring GPU-enabled version."""
    global _wo_focus
    if _wo_focus is None:
        try:
            # Try GPU-enabled waveorder first (has device param + batched FFT)
            if os.path.isdir(_GPU_WAVEORDER_PATH):
                import importlib

                _saved_path = sys.path.copy()
                sys.path.insert(0, _GPU_WAVEORDER_PATH)
                try:
                    # Force reimport from GPU path
                    if "waveorder.focus" in sys.modules:
                        del sys.modules["waveorder.focus"]
                    if "waveorder" in sys.modules:
                        del sys.modules["waveorder"]
                    from waveorder.focus import focus_from_transverse_band

                    # Verify it has the device parameter
                    sig = inspect.signature(focus_from_transverse_band)
                    if "device" in sig.parameters:
                        _wo_focus = focus_from_transverse_band
                        print(
                            f"[waveorder] Using GPU-enabled waveorder from {_GPU_WAVEORDER_PATH}"
                        )
                    else:
                        raise ImportError("GPU waveorder missing device param")
                except Exception:
                    sys.path[:] = _saved_path
                    raise
            if _wo_focus is None:
                raise ImportError("GPU waveorder not available")
        except (ImportError, Exception):
            # Fall back to installed waveorder
            try:
                from waveorder.focus import focus_from_transverse_band

                _wo_focus = focus_from_transverse_band
            except ImportError:
                pass
    return _wo_focus


class TransferFunctionSettings(BaseModel):
    wavelength_illumination: float
    yx_pixel_size: float
    z_pixel_size: float
    z_padding: int
    index_of_refraction_media: float
    numerical_aperture_detection: float
    numerical_aperture_illumination: float
    invert_phase_contrast: bool
    z_focus_offset: Optional[float] = None
    tilt_angle_zenith: Optional[float] = None
    tilt_angle_azimuth: Optional[float] = None


class ApplyInverseSettings(BaseModel):
    reconstruction_algorithm: str
    regularization_strength: float
    TV_rho_strength: float
    TV_iterations: int


class PhaseSettings(BaseModel):
    transfer_function: TransferFunctionSettings
    apply_inverse: ApplyInverseSettings


class ReconstructionConfig(BaseModel):
    input_channel_names: List[str]
    time_indices: str
    reconstruction_dimension: int
    phase: PhaseSettings


def model_to_yaml(model, yaml_path: Path) -> None:
    """
    Save a model's dictionary representation to a YAML file.

    Parameters
    ----------
    model : object
        The model object to convert to YAML.
    yaml_path : Path
        The path to the output YAML file.

    Raises
    ------
    TypeError
        If the `model` object does not have a `dict()` method.

    Notes
    -----
    This function converts a model object into a dictionary representation
    using the `dict()` method. It removes any fields with None values before
    writing the dictionary to a YAML file.

    Examples
    --------
    >>> from my_model import MyModel
    >>> model = MyModel()
    >>> model_to_yaml(model, 'model.yaml')

    """
    yaml_path = Path(yaml_path)

    if not hasattr(model, "dict"):
        raise TypeError("The 'model' object does not have a 'dict()' method.")

    model_dict = model.dict()

    # Remove None-valued fields
    clean_model_dict = {
        key: value for key, value in model_dict.items() if value is not None
    }

    with open(yaml_path, "w+") as f:
        yaml.dump(clean_model_dict, f, default_flow_style=False, sort_keys=False)


def yaml_to_model(yaml_path: Path, model):
    """
    Load model settings from a YAML file and create a model instance.

    Parameters
    ----------
    yaml_path : Path
        The path to the YAML file containing the model settings.
    model : class
        The model class used to create an instance with the loaded settings.

    Returns
    -------
    object
        An instance of the model class with the loaded settings.

    Raises
    ------
    TypeError
        If the provided model is not a class or does not have a callable constructor.
    FileNotFoundError
        If the YAML file specified by `yaml_path` does not exist.

    Notes
    -----
    This function loads model settings from a YAML file using `yaml.safe_load()`.
    It then creates an instance of the provided `model` class using the loaded settings.

    Examples
    --------
    # >>> from my_model import MyModel
    # >>> model = yaml_to_model('model.yaml', MyModel)

    """
    yaml_path = Path(yaml_path)

    if not callable(getattr(model, "__init__", None)):
        raise TypeError(
            "The provided model must be a class with a callable constructor."
        )

    try:
        with open(yaml_path, "r") as file:
            raw_settings = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The YAML file '{yaml_path}' does not exist.")

    return model(**raw_settings)


# Capability flags for sub-pixel focus support detection
_SUBPIXEL_CAPABLE: bool | None = None
_SUBPIXEL_WARNED: bool = False


def _normalize_subpixel_options(
    enable_subpixel_precision: bool,
    polynomial_fit_order: int | None,
) -> tuple[bool, int | None, dict[str, bool]]:
    """Normalize sub-pixel options based on Waveorder's focus API capabilities.

    Returns (enabled, order, support_map) where support_map indicates which kwargs are supported.
    Emits a one-time warning if requested but unsupported.
    """
    global _SUBPIXEL_CAPABLE, _SUBPIXEL_WARNED

    supported = {"enable_subpixel_precision": False, "polynomial_fit_order": False}
    wo_focus = _get_wo_focus()
    if wo_focus is None:
        # No focus function available at all; caller will raise ImportError
        return False, None, supported

    try:
        sig = inspect.signature(wo_focus)
        params = sig.parameters
        supported["enable_subpixel_precision"] = "enable_subpixel_precision" in params
        supported["polynomial_fit_order"] = "polynomial_fit_order" in params
        _SUBPIXEL_CAPABLE = supported["enable_subpixel_precision"]
    except Exception:
        _SUBPIXEL_CAPABLE = False

    if not enable_subpixel_precision:
        return False, None, supported

    if not _SUBPIXEL_CAPABLE:
        if not _SUBPIXEL_WARNED:
            _SUBPIXEL_WARNED = True
            try:
                import waveorder as _wo  # type: ignore

                ver = getattr(_wo, "__version__", "unknown")
            except Exception:
                ver = "unknown"
            print(
                f"[Auto2D][WARN] Sub-pixel focus requested but unsupported by this Waveorder version ({ver}). Falling back to integer focus."
            )
        return False, None, supported

    # Capable: ensure a sensible default order
    order = int(polynomial_fit_order) if polynomial_fit_order is not None else 2
    return True, order, supported


def is_valid_transfer_function_store(store_path: Path) -> bool:
    """Return True if `store_path` appears to be a valid OME-Zarr TF store.

    A store is considered valid if the directory exists and contains at least one
    array with a `.zarray` file either directly within the store or one level deeper.
    This is resilient to minor layout differences across Waveorder versions.
    """
    try:
        p = Path(store_path)
        if not p.exists() or not p.is_dir():
            return False
        for child in p.iterdir():
            if (child / ".zarray").exists():
                return True
            if child.is_dir():
                for grand in child.iterdir():
                    if (grand / ".zarray").exists():
                        return True
        return False
    except Exception:
        return False
