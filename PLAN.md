# PLAN — CrispEmbed integration into BiblioForge

Goal: use the sibling project **CrispEmbed** (a self-contained C++/ggml document-
understanding engine with Python `ctypes` bindings) to improve BiblioForge's OCR,
preprocessing, and metadata extraction — **without** adding heavyweight PyTorch/
transformers dependencies. Everything is wired as an **optional, soft-imported
backend** so BiblioForge keeps working when CrispEmbed isn't built.

## Why CrispEmbed

- Torch-free, GGUF-quantized, Metal/CUDA/Vulkan-accelerated.
- Ships working Python bindings: `../CrispEmbed/python/crispembed/_binding.py`
  (loaded via `_find_lib()`, which searches `../CrispEmbed/build/`).
- Exposes ready classes: `CrispScanCleanup`, `CrispOcrOrchestrator`,
  `CrispOcrModel`/`CrispOcrPipeline`, `CrispTextDetect`, `CrispLayout`,
  `CrispNER`, `CrispKIE`, `CrispLiLT`, `CrispTextLID`, plus 8 super-res engines
  and image-restoration models.

## Integration points in BiblioForge

- `extraction_manager.py` — `ExtractionManager._get_extractor()` (backend factory)
  and `ExtractionManager.extract()` (dispatch). New OCR backend plugs in here.
- `document_processor.py` — `DocumentProcessor`; metadata extraction
  (`extract_metadata_with_nanonets_ocr2`, `extract_metadata_with_docstrange`).
  New NER/KIE metadata path mirrors these.
- `BiblioForge.py` — argparse; new `--use-crispembed*` flags follow the existing
  `--use-nanonets-ocr2` / `--use-docstrange` pattern.
- `utils.py` — shared helpers (image rendering, import cache).
- `llm_providers.py` — model cache dir convention (shared with CrispEmbed GGUF cache).

---

## Phase 0 — Provisioning the native library  ✅ DONE

The dylib BiblioForge loads must be a **fresh, loadable** build.

- [x] Diagnosed: local `build/` dylib was stale (v0.7.0).
- [x] Diagnosed CI gap: `build.yml` uploaded only `libcrispembed.dylib` — no ggml
      runtime libs, and absolute runner rpaths → can't load on a consumer machine.
- [x] Built a fresh, complete local Release build (Metal + Apple BLAS) →
      `../CrispEmbed/build/libcrispembed.dylib` + matching `libggml*.dylib`. Loads OK.
- [x] **Fixed CI properly** (CrispEmbed PR #23, merged to `main` as `618b10c`):
      `cmake --install` stages the lib with its ggml deps and relocatable
      `@loader_path`/`$ORIGIN` rpath; artifacts now bundle the full `lib/`+`bin/` set.
      Verified: downloaded CI macos-arm64 artifact bundles all `libggml*.dylib`,
      rpath `@loader_path`, and loads standalone from a relocated dir.
- [ ] (Optional follow-up) apply the same install-bundle fix to the
      `build-android` job and static iOS link for full mobile parity.

## Phase 1 — Optional CrispEmbed adapter (foundation)  ✅ DONE

Single soft-import shim so the rest of the integration has one entry point.

- [x] Added `crispembed_adapter.py`:
  - Locates the sibling binding: prepends `$CRISPEMBED_PYTHON_DIR` or
    `../CrispEmbed/python` to `sys.path`, then imports `crispembed._binding` and
    force-loads the dylib so a broken native lib is caught at probe time.
  - `is_available()` / `unavailable_reason()`; result cached (probe once).
  - Lazy accessors: `get_scan_cleanup()`, `get_class(name)`; clear error pointing
    at PLAN Phase 0 when unavailable.
- [x] Smoke-tested: `is_available()` True with the fresh build; scan cleanup runs.
- [ ] (Later) honor the shared GGUF cache dir for model-backed engines
      (reuse `_ensure_cache_dir` from `llm_providers.py`, incl. external-SSD symlink).

## Phase 2 — Pre-OCR scan cleanup  ✅ DONE (highest value / lowest risk)

Runs `CrispScanCleanup` (classical, no model) on each page image *before* the
existing OCR engine. Improves every OCR method, adds no heavy deps.

- [x] Added `--scan-cleanup {off,auto,on}` (default `auto`) and
      `--scan-cleanup-binarize {off,otsu,sauvola}` to `BiblioForge.py`; build a
      `scan_cleanup_config` and thread it through `process_files` →
      `_process_single_file` → `manager.extract` (kwargs).
- [x] `ExtractionManager.extract` sets it on the PDF extractor via
      `set_scan_cleanup()` (guarded by `hasattr`).
- [x] `PDFExtractor`: `set_scan_cleanup()`, lazy `_get_scan_cleaner()` (soft import
      of the adapter, fails quietly once), and `_apply_scan_cleanup(pil)→pil`
      (never raises). Inserted into the per-page loops of tesseract, easyocr,
      paddleocr, doctr, and kraken(API). kraken_cli (disk-based) left as-is.
- [x] Verified: disabled → identity; enabled → logs activation, returns valid
      cleaned image; OCR on page 30 of `changingprofileo0000crow_1.pdf` = 441 words
      cleaned vs 441 original (no regression on an already-clean page). Normal
      text-layer run still exits 0.
- [ ] (Later) `auto` heuristic to skip cleanup on clearly-clean pages; tune crop
      thresholds; measure gains on a deliberately skewed/noisy sample.

## Phase 3 — Auto-DPI + super-resolution for low-res scans  ✅ DONE (wired; see caveat)

- [x] Adapter `get_super_resolver(engine)` → `.upscale(pil)→pil` wrapper over the
      SR engines (pan/swinir/hat/esrgan/safmn), with registry auto-download.
- [x] `--sr {off,auto,on}`, `--sr-engine`, `--sr-min-width`; threaded through to
      `PDFExtractor.set_super_resolution()`.
- [x] `_apply_super_resolution()` (auto = upscale pages narrower than min_width;
      safety cap `max_input_width=2500` so huge pages are never upscaled) chained
      before scan cleanup in `_preprocess_ocr_image()`, used by all OCR loops.
      Logs activation once; never raises.
- [x] Verified trigger logic: off→identity; small(400)→1600 upscaled; big(3000)
      and med(1800)→skipped.
- [ ] **Caveat / follow-up:** BiblioForge rasterizes at a fixed 300 DPI, so pages
      are almost always wider than the cap and SR rarely triggers in the normal
      pipeline. To make SR genuinely useful, render scanned PDFs at their *native*
      embedded DPI (via `pdf_info`) and SR-upscale from there. Default is `off`.

## Known issues & mitigations (discovered during 1–3)

- **ggml Metal teardown abort at process exit.** When the extractors stack (which
  imports torch/MPS) and a CrispEmbed **Metal** engine are both loaded, the C++
  static-destructor teardown aborts (SIGABRT/exit 134) *after* all work + output.
  CPU-only engines (e.g. `CrispScanCleanup`) are unaffected. **Mitigation:** the
  adapter flags when any native engine loads (`native_engine_loaded()`), and
  `BiblioForge.py`'s `__main__` does a flushed `os._exit(code)` in that case,
  bypassing the buggy teardown. Verified: exit 0 with Metal SR loaded.
  *Upstream follow-up:* fix the Metal device global destructor in ggml/CrispEmbed.
- **CrispEmbed downloader doesn't create its cache dir.** `resolve_model` fails if
  the cache dir is missing (curl can't write the `.tmp`). The adapter's
  `_resolve_cache_dir()` mkdirs the chosen dir (and honors the external-SSD
  symlink with local fallback) before any download. *Upstream follow-up:* mkdir
  in `crispembed_mgr` before download.

## Phase 4 — CrispEmbed as an OCR backend  🟡 CODE COMPLETE, validation pending

Torch-free alternative to the nanonets/docstrange VLM backends. Implemented as a
new OCR method inside `PDFExtractor` (reuses rasterization + `_preprocess_ocr_image`),
using a **single-pass** `CrispOcrModel.recognize(page)` — one forward pass per page.

- [x] `--ocr-method crispembed` + `--crispembed-ocr-model` (default `got-ocr2`),
      `--crispembed-ocr-dpi` (150), `--crispembed-ocr-cpu`. Config threaded through
      to `PDFExtractor.set_crispembed_ocr()`.
- [x] Adapter `get_ocr_model(name, force_cpu)`; `PDFExtractor.extract_with_crispembed()`,
      `_init_ocr('crispembed')` (lazy load + download), opt-in availability
      (`_is_method_available` only True when configured + CrispEmbed present so it
      never joins the auto-fallback chain unasked).
- [ ] **Validation pending:** blocked on slow download of a full-page OCR model
      (`got-ocr2`, 750 MB + vision projector). Test recognize() on real pages and
      compare vs tesseract once the model is local.
- [ ] **Architecture note (learned):** the small DBNet+TrOCR pipeline
      (`CrispOcrPipeline`) is a poor fit — it aborts on Metal (`unsupported op
      'CPY'` in DBNet) and is too slow per-region on CPU. Hence single-pass VLM
      (`CrispOcrModel`) instead. `--crispembed-ocr-cpu` sets the per-engine
      FORCE_CPU envs as an escape hatch if a VLM also hits a Metal op gap.

## Phase 5 — Local metadata via NER / language ID  ✅ DONE & VERIFIED

Cheaper, offline alternative (or pre-pass) to the LLM metadata step.

- [x] `--metadata-backend {llm,crispembed-ner,hybrid}` (default `llm`), threaded to
      `_process_single_file`.
- [x] Adapter `get_ner('gliner-lfm-q4k')` + `get_lid('glotlid')`.
- [x] `DocumentProcessor.extract_metadata_with_crispembed_ner(text)`: runs LID for
      language + zero-shot GLiNER over the leading title-page text for
      author/title/year/publisher → `{title,author,year,language}` (4-digit-year
      regex fallback; author run through `sort_author_with_retries`).
- [x] Wired into the sort flow: `crispembed-ner` = NER only; `hybrid` = NER then
      LLM fallback; `llm` = unchanged. Soft-fails to None when CrispEmbed absent.
- [x] **Verified** on `changingprofileo0000crow_1.txt`: produced author=`MICHAEL
      BERTRAM CROWE`, year=`1977`, language=`en`, title extracted; full `--sort
      --metadata-backend crispembed-ner` CLI run generated the correct rename
      script with no LLM/Ollama and exit 0.
- [x] Model notes: `gliner-lfm-q4k` is license-restricted (LFM1.0) and `glotlid`'s
      URL 404s — defaults switched to **`gliner-deberta-q4k`** (152 MB) and
      **`cld3`** for LID, both of which download cleanly.
- [x] Fixed gating: `crispembed-ner` skips LLM-provider init so a missing Ollama
      no longer disables sorting; `hybrid` tolerates LLM-init failure (NER-only).
- [ ] (Later) `CrispKIE`/`CrispLiLT` for layout-aware structured fields; true
      hybrid that seeds the LLM prompt with NER spans; author-name sorting without
      an LLM (currently kept as-is when no provider).

## Phase 6 — Polish  (todo)

- [ ] Optional layout detection (`CrispLayout`) to route tables/formulas.
- [ ] Optional image restoration (NAFNet/Restormer) for degraded scans.
- [ ] README: enabling CrispEmbed, the GGUF cache + external-SSD convention,
      `--scan-cleanup` / `--metadata-backend` usage.
- [ ] Upstream CrispEmbed fixes: cache-dir mkdir before download; Metal device
      teardown abort; Android/iOS install-bundle parity.

## Phase 6 — Polish

- [ ] Optional: layout detection (`CrispLayout`, 17 classes) to segment pages /
      route tables & formulas to specialized handling.
- [ ] Optional: image restoration (NAFNet/Restormer) for degraded scans.
- [ ] Docs: README section on enabling CrispEmbed; note the GGUF model cache and
      the external-SSD symlink convention.
- [ ] Ensure all new backends remain fully optional (no import → graceful skip).

---

## Conventions / notes

- All CrispEmbed use goes through `crispembed_adapter.py` — no direct `_binding`
  imports scattered across modules.
- Every new capability is **off-by-default or `auto`**, gated by a flag, and a
  no-op when CrispEmbed isn't installed.
- GGUF models are large; store under the shared cache dir (external SSD when
  mounted — see `_ensure_cache_dir` in `llm_providers.py`). Never silently
  truncate or skip — `log()` what was done.
- Integrate **one phase at a time**, verifying on real PDFs before the next.
