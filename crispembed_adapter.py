"""Optional adapter for the sibling CrispEmbed engine.

CrispEmbed (https://github.com/CrispStrobe/CrispEmbed) is a self-contained
C++/ggml document-understanding engine with Python ctypes bindings. BiblioForge
uses it — when available — for pre-OCR scan cleanup, super-resolution, a
torch-free OCR backend, and local NER/KIE metadata extraction.

Everything here is OPTIONAL. If CrispEmbed isn't built/importable, `is_available()`
returns False and callers must fall back to BiblioForge's existing behavior.
All CrispEmbed access in BiblioForge goes through this module — no other file
imports `crispembed._binding` directly.

Resolution order for the binding directory:
  1. $CRISPEMBED_PYTHON_DIR (explicit override)
  2. ../CrispEmbed/python  (sibling checkout, the common dev layout)
  3. an installed `crispembed` package on sys.path
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level caches so we probe at most once.
_BINDING = None              # the imported crispembed._binding module (or None)
_AVAILABLE: Optional[bool] = None
_LOAD_ERROR: Optional[str] = None
# Set once any native engine is instantiated. ggml's Metal backend can abort
# during C++ static-destructor teardown (a known torch-MPS + ggml-Metal exit
# interaction), so the app bypasses that teardown with os._exit when this is set.
_NATIVE_ENGINE_LOADED = False


def native_engine_loaded() -> bool:
    """True if any CrispEmbed native engine was instantiated this process."""
    return _NATIVE_ENGINE_LOADED


def _note_engine_loaded():
    global _NATIVE_ENGINE_LOADED
    _NATIVE_ENGINE_LOADED = True


def _resolve_cache_dir():
    """Pick a usable CrispEmbed model cache dir and export CRISPEMBED_CACHE_DIR.

    Honors the user's external-SSD-with-fallback convention: the preferred path
    ~/.cache/crispembed is typically a symlink to an external volume. When the
    SSD is mounted we use it; when it isn't (broken symlink) we fall back to a
    local dir. We always mkdir the chosen target — CrispEmbed's downloader does
    not create the cache dir itself and fails if it's missing.

    Respects an explicit CRISPEMBED_CACHE_DIR if the user already set one.
    """
    if os.environ.get("CRISPEMBED_CACHE_DIR"):
        d = Path(os.environ["CRISPEMBED_CACHE_DIR"]).expanduser()
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return d

    candidates = [
        Path.home() / ".cache" / "crispembed",         # preferred (often a symlink to external SSD)
        Path.home() / ".cache" / "crispembed_local",   # local fallback when the SSD is absent
    ]
    for cand in candidates:
        try:
            if cand.is_symlink():
                # Resolve manually: mkdir won't create through a symlink whose
                # target doesn't exist yet. Usable only if the target's parent
                # (the mount point) is present.
                target = Path(os.readlink(cand))
                if not target.is_absolute():
                    target = (cand.parent / target).resolve()
                if not target.parent.exists():
                    raise FileNotFoundError(f"symlink target unavailable: {target}")
                target.mkdir(parents=True, exist_ok=True)
            else:
                cand.mkdir(parents=True, exist_ok=True)
            os.environ["CRISPEMBED_CACHE_DIR"] = str(cand)
            return cand
        except OSError as e:
            logger.debug("CrispEmbed cache dir unavailable: %s (%s)", cand, e)
            continue
    return None


def _candidate_python_dirs():
    """Yield directories that may contain the `crispembed` python package."""
    env = os.environ.get("CRISPEMBED_PYTHON_DIR")
    if env:
        yield Path(env).expanduser()
    # Sibling checkout relative to this file: <repo>/../CrispEmbed/python
    here = Path(__file__).resolve().parent
    yield here.parent / "CrispEmbed" / "python"


def _load_binding():
    """Import crispembed._binding and verify the native lib actually loads.

    Returns the binding module on success, or None (recording the reason in
    _LOAD_ERROR). Cached via _AVAILABLE.
    """
    global _BINDING, _AVAILABLE, _LOAD_ERROR
    if _AVAILABLE is not None:
        return _BINDING

    for d in _candidate_python_dirs():
        if d.is_dir() and str(d) not in sys.path:
            sys.path.insert(0, str(d))

    # Point CrispEmbed's model downloader at a writable cache before first use.
    _resolve_cache_dir()

    try:
        from crispembed import _binding as binding  # type: ignore
        # Importing the module is not enough — the dylib loads lazily. Force it
        # so a missing/broken native library is detected here, not mid-pipeline.
        binding._load_library()
    except Exception as e:  # ImportError, OSError (dlopen), etc.
        _LOAD_ERROR = f"{type(e).__name__}: {e}"
        _AVAILABLE = False
        _BINDING = None
        logger.debug("CrispEmbed unavailable: %s", _LOAD_ERROR)
        return None

    _BINDING = binding
    _AVAILABLE = True
    logger.debug("CrispEmbed binding loaded from %s", getattr(binding, "__file__", "?"))
    return binding


def is_available() -> bool:
    """True if the CrispEmbed binding and its native library load successfully."""
    return _load_binding() is not None


def unavailable_reason() -> Optional[str]:
    """Human-readable reason CrispEmbed couldn't be loaded (None if available)."""
    _load_binding()
    return _LOAD_ERROR


def _require():
    binding = _load_binding()
    if binding is None:
        raise RuntimeError(
            "CrispEmbed is not available ({}). Build it (cmake --build build in "
            "the CrispEmbed checkout) or set CRISPEMBED_PYTHON_DIR. See PLAN.md "
            "Phase 0.".format(_LOAD_ERROR or "not found")
        )
    return binding


# --- Engine accessors -------------------------------------------------------
# Thin lazy factories. Each raises a clear error when CrispEmbed is missing.

def get_scan_cleanup(**kwargs):
    """Return a CrispScanCleanup instance (classical, no model needed)."""
    inst = _require().CrispScanCleanup(**kwargs)
    _note_engine_loaded()
    return inst


def get_class(name: str):
    """Return a CrispEmbed binding class by name (e.g. 'CrispOcrOrchestrator',
    'CrispNER', 'CrispTextLID', 'CrispPanSr'). Raises if missing."""
    binding = _require()
    cls = getattr(binding, name, None)
    if cls is None:
        raise AttributeError(f"CrispEmbed has no '{name}' in this build")
    return cls


def resolve_model(name: str, auto_download: bool = True) -> str:
    """Resolve a registry model name to a local GGUF path, downloading if needed."""
    return _require().CrispEmbed.resolve_model(name, auto_download=auto_download)


# --- Full-page OCR (single-pass models via CrispOcrModel.recognize) ---------
# Force-CPU env vars per VLM OCR engine, used when force_cpu is requested to
# dodge ggml Metal unsupported-op aborts (e.g. some models hit 'CPY' on Metal).
_OCR_FORCE_CPU_ENVS = (
    "GLM_OCR_FORCE_CPU", "QWEN2VL_OCR_FORCE_CPU", "INTERNVL2_OCR_FORCE_CPU",
    "MATH_OCR_FORCE_CPU", "PARSEQ_OCR_FORCE_CPU", "MIXTEX_OCR_FORCE_CPU",
    "PPFNL_OCR_FORCE_CPU", "OCR_DETECT_FORCE_CPU",
)


def get_ocr_model(model_name: str = "got-ocr2", force_cpu: bool = False):
    """Resolve+load a single-pass OCR model. Returns a CrispOcrModel with a
    .recognize(image) -> str method (image may be a PIL Image / path / ndarray)."""
    if force_cpu:
        for v in _OCR_FORCE_CPU_ENVS:
            os.environ.setdefault(v, "1")
    path = resolve_model(model_name)
    inst = get_class("CrispOcrModel")(path)
    _note_engine_loaded()
    return inst


# --- NER / language identification (local metadata) -------------------------
_ner_cache = {}
_lid_cache = {}


def get_ner(model_name: str = "gliner-deberta-q4k"):
    """Zero-shot NER (GLiNER): .extract(text, labels, threshold) -> list of
    {text,label,start,end,score}. Cached per model."""
    if model_name not in _ner_cache:
        path = resolve_model(model_name)
        _ner_cache[model_name] = get_class("CrispNER")(path)
        _note_engine_loaded()
    return _ner_cache[model_name]


def get_lid(model_name: str = "cld3"):
    """Language identification: .predict(text) -> (iso_code, confidence). Cached."""
    if model_name not in _lid_cache:
        path = resolve_model(model_name)
        _lid_cache[model_name] = get_class("CrispTextLID")(path)
        _note_engine_loaded()
    return _lid_cache[model_name]


# --- Super-resolution -------------------------------------------------------
# Map a short engine id to its registry model + binding class. All SR engines
# share the process(pixels, width, height) -> (ndarray, w, h) signature.
_SR_MODELS = {
    "pan": "pan-x4", "swinir": "swinir-sr-x4", "hat": "hat-sr-x4",
    "esrgan": "esrgan-x4", "safmn": "safmn-x4",
}
_SR_CLASSES = {
    "pan": "CrispPanSr", "swinir": "CrispSwinirSr", "hat": "CrispHatSr",
    "esrgan": "CrispEsrganSr", "safmn": "CrispSafmnSr",
}
_sr_cache = {}


def get_super_resolver(engine: str = "swinir"):
    """Return a cached super-resolver with an .upscale(pil_image)->pil_image method."""
    if engine not in _SR_MODELS:
        raise ValueError(f"Unknown SR engine '{engine}'. Options: {list(_SR_MODELS)}")
    if engine not in _sr_cache:
        model = resolve_model(_SR_MODELS[engine])
        impl = get_class(_SR_CLASSES[engine])(model)
        _note_engine_loaded()
        _sr_cache[engine] = _SuperResolver(impl)
    return _sr_cache[engine]


class _SuperResolver:
    def __init__(self, impl):
        self._impl = impl

    def upscale(self, pil_image):
        import numpy as np
        from PIL import Image
        arr = np.asarray(pil_image.convert("RGB"))
        h, w = arr.shape[:2]
        result = self._impl.process(arr, w, h)
        out = result[0] if isinstance(result, tuple) else result
        return Image.fromarray(out)
