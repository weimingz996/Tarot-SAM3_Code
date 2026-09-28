# Candidate-Only Open-Source Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Tarot-SAM3 return pre-vote candidate mask records while removing final voting and downstream MSR implementation from the public repository.

**Architecture:** Keep the existing Reason and Refer candidate generators, but end each flow at the exact input boundary of its former final vote. Reason exposes Text Top3 plus FullDes through a candidate-only runner; Refer returns its existing `masks_info`. The CLI serializes every top-level candidate independently, and repository-level tests prevent final-vote or MSR code from returning.

**Tech Stack:** Python 3.11, NumPy, OpenCV, PyYAML, `unittest`, SAM3, Qwen API client

**Spec:** `docs/superpowers/specs/2026-09-28-candidate-only-open-source-boundary-design.md`

## Global Constraints

- `TarotSAM3.process_image(...)` returns `list[dict]`; every record has `id`, `source`, and a two-dimensional boolean `mask`.
- Preserve empty top-level candidate masks and existing candidate metadata.
- Reason returns selectable Text Top3 and FullDes records only; bbox masks remain archive evidence, not selectable outputs.
- Refer keeps the bbox representative selection needed to construct `masks_info`, but removes the final cross-candidate vote.
- Do not replace removed voting or MSR with another selection heuristic.
- Do not add dependencies or new configuration layers.
- The source tree must not contain the removed final-vote or MSR implementation.
- The CLI writes candidate PNGs under `<save_dir>/candidate_masks/` and never writes `final_output_mask.png`.
- README environment snapshot: Python 3.11.14; torch 2.8.0; torchvision 0.23.0; transformers 4.57.0; accelerate 1.11.0; numpy 1.26.4; opencv-python 4.10.0.84; Pillow 10.4.0; scipy 1.16.3; scikit-learn 1.3.2; matplotlib 3.10.7; seaborn 0.13.2; PyYAML 6.0.2; requests 2.32.3; tqdm 4.66.5; triton 3.4.0.

## Review Focus

- Candidate-generation exceptions propagate unchanged and never trigger a fallback selector; tested in Task 2.
- Empty candidate sets return `[]`, while candidate records with empty masks remain present; tested in Tasks 1 and 2.
- Reason bbox evidence remains archived but never appears in `candidate_masks_info`; tested in Task 1.
- Refer preserves nested `raw_masks_info` and the selected bbox representative in its top-level list; tested in Task 2.
- Repeated or unsafe candidate names produce deterministic, non-colliding PNG paths because the numeric prefix is authoritative; tested in Task 4.

---

### Task 1: Cut Reason at the Candidate Boundary

**Files:**
- Create: `tests/test_reason_candidate_boundary.py`
- Modify: `reason_prompt/reasonseg_best_flow_reference.py:1-3050`
- Modify: `reason_prompt/reasonseg_best_flow.py:1-380`
- Delete: `reason_prompt/reasonseg_shortlong_bbox_experiment.py`

**Interfaces:**
- Consumes: Existing FullDes generation, `generate_text_augmentation(...)`, `run_text_masks(...)`, `run_bbox_augmentation(...)`, and `save_candidate_archive(...)`.
- Produces: `run_reasonseg_candidate_flow(*, full_des_qwen, text_qwen, sam3, image_path, query, output_dir, sample="live", visualize=False) -> dict[str, Any]`. The returned mapping contains `candidate_masks_info: list[dict]`, `full_des: dict`, `result_path: str`, and the persisted candidate-only result metadata.
- Produces: `_candidate_masks_info(text_top3, full_des_mask, full_result) -> list[dict]`, with Text Top records followed by one FullDes record.

- [ ] **Step 1: Write the failing candidate-record test**

Add `test_candidate_masks_info_preserves_empty_masks_and_excludes_bbox_evidence`. Use one non-empty Text Top mask, one empty Text Top mask, one FullDes mask, and unrelated bbox metadata. Assert the returned IDs are `Text Top1`, `Text Top2`, `FullDes`; every mask is boolean and retains its input pixels; no bbox record appears; and existing text metadata survives.

- [ ] **Step 2: Write the failing source-boundary test**

Add `test_reason_source_contains_no_final_vote_implementation`. Assert:

```python
assert not Path("reason_prompt/reasonseg_shortlong_bbox_experiment.py").exists()
source = Path("reason_prompt/reasonseg_best_flow_reference.py").read_text()
for forbidden in ("def select_best_mask(", "def training_free_vote(", "def refine_vote("):
    assert forbidden not in source
```

Also assert `reasonseg_best_flow.py` does not import `reasonseg_shortlong_bbox_experiment`.

- [ ] **Step 3: Run the focused tests to verify RED**

Run: `python -m unittest tests.test_reason_candidate_boundary -v`

Expected: FAIL because `_candidate_masks_info` and `run_reasonseg_candidate_flow` do not exist and the final-vote source still exists.

- [ ] **Step 4: Implement `_candidate_masks_info(...)`**

In `reasonseg_best_flow_reference.py`, add the exact signature from Interfaces. Reuse the existing Text Top metadata and add `id`, `source`, and boolean `mask`; add a final `FullDes` record with `source="full_des"`. Do not inspect bbox candidates and do not filter empty masks.

- [ ] **Step 5: Replace `run_sample(...)` with candidate-only persistence**

Rename it to `run_candidate_sample(image_path: Path, query: str, output_dir: Path, engines: Engines) -> dict[str, Any]`. Keep stages through bbox generation and `save_candidate_archive`, remove `select_best_mask`, selected-mask PNG creation, and selection metadata. Persist schema `reasonseg-candidates-v1`, set `STAGE_ORDER` to `("full_des", "text_augmentation", "bbox_augmentation", "candidate_export")`, and seal the result with the candidate archive hash rather than a selected-mask hash. Return the persisted result plus `candidate_masks_info`, `full_des`, and `result_path` in memory.

- [ ] **Step 6: Remove final-vote functions from the Reason reference module**

Delete the contiguous final-vote implementation beginning at `unique_winner(...)` and ending before `orient_mask_to_image(...)`. Preserve candidate-generation helpers through `run_bbox_augmentation(...)` and persistence helpers beginning with `orient_mask_to_image(...)`. Remove imports used only by the deleted block.

- [ ] **Step 7: Simplify the public Reason runner**

Remove `run_reasonseg_best_flow(...)` and add the candidate-only signature in Interfaces. It constructs `core.Engines` and delegates directly to `core.run_candidate_sample(...)`. Remove the Short/Long runner, A3.x constants, validation functions, selected-mask writing, and experiment import. Keep `VisionQwen.from_qwen_config(...)`.

- [ ] **Step 8: Delete the Reason Short/Long vote module**

Delete `reason_prompt/reasonseg_shortlong_bbox_experiment.py`; do not move any selection logic elsewhere.

- [ ] **Step 9: Run the focused tests to verify GREEN**

Run: `python -m unittest tests.test_reason_candidate_boundary -v`

Expected: PASS.

- [ ] **Step 10: Commit the Reason boundary**

```bash
git add tests/test_reason_candidate_boundary.py reason_prompt/reasonseg_best_flow.py reason_prompt/reasonseg_best_flow_reference.py
git commit -m "refactor: expose Reason candidate masks before voting"
```

### Task 2: Change the Main API and Refer Flow to Return Candidates

**Files:**
- Create: `tests/test_candidate_only_pipeline.py`
- Modify: `tarot_sam3.py:1-1080`
- Delete: `ref_prompt/mask_candidate_selector.py`

**Interfaces:**
- Consumes: Task 1's `run_reasonseg_candidate_flow(...) -> dict[str, Any]` and its `candidate_masks_info` field.
- Produces: `TarotSAM3.process_image(image_path: str, query: str, reason_seg: bool, logger: ExperimentLogger, save_dir: str, visualize: bool = True) -> list[dict]`.
- Produces: `TarotSAM3.expression_reasoning_interpreter() -> list[dict]`, returning the Refer `masks_info` immediately before the former final cross-candidate vote.

- [ ] **Step 1: Write the failing Reason main-flow test**

Add `test_reason_process_returns_runner_candidates_without_selecting`. Load `tarot_sam3.py` with lightweight `sys.modules` stubs for model-dependent imports so the test does not require Triton or model weights. Construct `TarotSAM3` with `__new__`, provide minimal fake `qwen`, `sam3`, logger, configuration, and a fake `reasonseg_candidate_runner` returning two candidate records including one empty mask. Patch image loading. Assert `process_image(..., reason_seg=True, ...)` returns the same two records in order and makes no call to any vote or MSR hook.

- [ ] **Step 2: Write the failing Refer boundary tests**

Add:

- `test_refer_interpreter_returns_prevote_masks_info`, stubbing description, bbox, filtering, and bbox representative helpers. Assert the result includes the text records, preserves nested `raw_masks_info`, and appends the selected bbox representative.
- `test_refer_process_returns_empty_candidate_list`, stubbing Refer candidate generation to return `[]`. Assert `process_image(...) == []` and the logger records the empty set without creating a mask.
- `test_candidate_generation_error_propagates`, making the mode runner raise `RuntimeError("candidate generation failed")` and asserting the same exception reaches the caller without invoking a fallback.

- [ ] **Step 3: Run the focused tests to verify RED**

Run: `python -m unittest tests.test_candidate_only_pipeline -v`

Expected: FAIL because `process_image` still returns one final mask and the Refer interpreter still calls `_consensus_winner`.

- [ ] **Step 4: Switch Tarot-SAM3 to the candidate runner**

Import `run_reasonseg_candidate_flow`, expose it as `reasonseg_candidate_runner`, and remove the old best-flow symbol. In the Reason branch of `process_image`, call the candidate runner, retain FullDes text/scope metadata, assign `self.masks_info` from `candidate_masks_info`, log every candidate, and return that list immediately.

- [ ] **Step 5: Stop Refer immediately before final consensus**

Change `expression_reasoning_interpreter()` to return `masks_info` after text/bbox construction and filtering. Remove winner extraction, `ERI_mask`, `msr_masks`, and full-description bbox fallback derived from a winner. In the Refer branch of `process_image`, return the interpreter's list immediately, including `[]`.

- [ ] **Step 6: Remove final vote and downstream execution from `tarot_sam3.py`**

Delete `_consensus_winner`, `_save_eri_masks_grid`, the entire MSR loop after candidate generation, and state used only by winner/MSR handling. Do not add a replacement ranking or preferred-candidate field.

- [ ] **Step 7: Delete the unused Refer final selector**

Delete `ref_prompt/mask_candidate_selector.py`. Keep `ref_prompt/training_free_bbox_selector.py`, which forms the bbox representative before the public boundary.

- [ ] **Step 8: Run the focused tests to verify GREEN**

Run: `python -m unittest tests.test_candidate_only_pipeline -v`

Expected: PASS.

- [ ] **Step 9: Commit the public API cutoff**

```bash
git add tests/test_candidate_only_pipeline.py tarot_sam3.py
git commit -m "refactor: return pre-vote candidate masks"
```

### Task 3: Remove MSR-Only Source and Configuration

**Files:**
- Create: `tests/test_public_source_boundary.py`
- Modify: `tarot_sam3.py`
- Modify: `src/models.py:1-230`
- Modify: `src/utils.py:1-760`
- Modify: `src/prompts.py:1-130`
- Modify: `configs/7B.yaml`

**Interfaces:**
- Consumes: Task 2's candidate-only `TarotSAM3` API.
- Produces: A source tree with no DINO/MSR runtime path or final-vote symbols; pre-vote candidate behavior remains unchanged.

- [ ] **Step 1: Write the failing repository-boundary test**

Add `test_removed_vote_and_msr_symbols_are_absent`. Scan production `.py` files and `configs/*.yaml`, excluding docs and tests, and assert none contains these implementation symbols:

```python
FORBIDDEN = (
    "reasonseg_shortlong_bbox_experiment",
    "def unique_winner(",
    "def stable_vote(",
    "def build_vote_feature(",
    "def select_best_mask(",
    "def training_free_vote(",
    "def _consensus_winner(",
    "def ERI_point_extractor(",
    "def mask_self_refine(",
    "class DINOv3Engine",
    "final_output_mask.png",
)
```

Add `test_config_contains_no_deleted_pipeline_sections`, asserting `dinov3` and `tarot_sam3.MSR` are absent and the ERI mapping contains none of the removed point-sampling or Reason vote keys.

- [ ] **Step 2: Run the boundary tests to verify RED**

Run: `python -m unittest tests.test_public_source_boundary -v`

Expected: FAIL on existing DINO/MSR code and configuration.

- [ ] **Step 3: Remove MSR methods, imports, and state from `tarot_sam3.py`**

Delete `ERI_point_extractor`, `output_checker`, `mask_comparator`, `extract_point_under`, `extract_point_over`, and `mask_self_refine`. Remove `tempfile`, DINO, point-mask, comparator, dilation, and visualization imports and instance fields that have no remaining callers. Keep pre-vote Refer prompt generation, `split_phases`, bbox utilities, and candidate visualization inputs.

- [ ] **Step 4: Remove the DINO model wrapper**

Delete `DINOv3Engine` from `src/models.py`; then remove `torchvision.transforms.functional` and `torch.nn.functional` only if repository search confirms they have no remaining use in that file. Keep Qwen, SAM3, bbox, and mask utilities used by candidate generation.

- [ ] **Step 5: Remove MSR-only utilities and prompts**

From `src/utils.py`, delete helpers whose only callers were the removed MSR path: point/similarity visualization, matrix combination, negative-point search, DINO point extraction, masked-image generation, dilation, centroid/gate/comparator helpers, mask-point sampling, disagreement visualization, resize, and largest-component comparison. Preserve `save_mask`, `split_phases`, bbox extraction/filtering, candidate boards, and logger/config loading.

From `src/prompts.py`, keep `sam3_multi_expression` and `ref_refine`; delete output checkers, part checkers, and mask comparator prompts.

- [ ] **Step 6: Remove deleted-pipeline configuration**

From `configs/7B.yaml`, delete the top-level `dinov3` mapping, the complete `MSR` mapping, Reason vote settings, and Reason/Refer sample-point settings. Keep Qwen, SAM3, Reason SAM confidence and Refer controls, bbox thresholds, Refer variant threshold, and `refer_reference_exact_copy_block_iou: 0.9`.

- [ ] **Step 7: Run boundary and behavior tests to verify GREEN**

Run: `python -m unittest tests.test_public_source_boundary tests.test_candidate_only_pipeline tests.test_reason_candidate_boundary -v`

Expected: PASS.

- [ ] **Step 8: Commit the MSR source removal**

```bash
git add tests/test_public_source_boundary.py tarot_sam3.py src/models.py src/utils.py src/prompts.py configs/7B.yaml
git commit -m "refactor: remove final vote and MSR implementation"
```

### Task 4: Save Candidate PNGs and Document the Public Release

**Files:**
- Create: `tests/test_candidate_cli_output.py`
- Modify: `tarot_sam3.py:1070-end`
- Modify: `src/utils.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 2's `list[dict]` candidate API and `src.utils.save_mask(...)`.
- Produces: `src.utils.save_candidate_masks(candidates: list[dict], save_dir: str, logger: ExperimentLogger) -> list[str]` and documented CLI/Python behavior.

- [ ] **Step 1: Write the failing candidate-file test**

Add `test_save_candidate_masks_uses_stable_unique_names`. Supply two records with the same unsafe ID `"Text / Candidate"`, one empty mask, and distinct sources. Assert the returned basenames are `01_text_candidate.png` and `02_text_candidate.png`, both files exist, and the logger receives source and pixel-count information.

Add `test_save_candidate_masks_accepts_empty_list`, asserting it creates an empty `candidate_masks` directory and returns `[]`.

- [ ] **Step 2: Run the CLI-output tests to verify RED**

Run: `python -m unittest tests.test_candidate_cli_output -v`

Expected: FAIL because `save_candidate_masks` does not exist.

- [ ] **Step 3: Implement `save_candidate_masks(...)`**

In `src/utils.py`, create `<save_dir>/candidate_masks`, normalize each mask to boolean, sanitize the lowercase candidate ID to `[a-z0-9_]+` with fallback `candidate`, prefix the one-based two-digit index, save through `save_mask`, log the record, and return save-dir-relative paths. The numeric prefix guarantees uniqueness; do not add collision-management state.

- [ ] **Step 4: Update the CLI entry point**

Rename `output_mask` to `candidates`, call `save_candidate_masks`, and remove `final_output_mask.png` handling. Print/log the number of generated candidates without selecting one.

- [ ] **Step 5: Rewrite the README for the candidate-only release**

Document:

- The repository stops before final best-mask voting and contains no MSR implementation.
- `process_image(...) -> list[dict]` with `id`, `source`, and boolean `mask`.
- CLI output under `candidate_masks/` and absence of a final prediction.
- Reason returns Text Top3 + FullDes; Refer returns its pre-consensus `masks_info`.
- Environment source `/hpc2hdd/home/wzhang915/.conda/envs/SAM123`, Python 3.11.14, and the exact 15-package table from the spec.

- [ ] **Step 6: Run the CLI-output tests to verify GREEN**

Run: `python -m unittest tests.test_candidate_cli_output -v`

Expected: PASS.

- [ ] **Step 7: Commit CLI and documentation changes**

```bash
git add tests/test_candidate_cli_output.py tarot_sam3.py src/utils.py README.md
git commit -m "docs: publish candidate-only output contract"
```

### Task 5: Full Verification and Release-Boundary Audit

**Files:**
- Modify only if verification exposes a defect in files already listed above.

**Interfaces:**
- Consumes: All prior task outputs.
- Produces: Fresh evidence that the candidate-only boundary is complete and internally consistent.

- [ ] **Step 1: Run the complete available test suite**

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS with zero errors and zero failures. If unrelated pre-existing tests are unavailable or fail because optional model dependencies are absent, record their exact names and continue with the isolated boundary suite; do not report the full suite as passing.

- [ ] **Step 2: Compile every public Python module without writing workspace bytecode**

Run: `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/tarot-candidate-pycache.XXXXXX) python -m compileall -q tarot_sam3.py reason_prompt ref_prompt src sam3 tests`

Expected: exit code 0 and no output.

- [ ] **Step 3: Scan for forbidden production symbols and deleted files**

Run production-only `rg` checks for the `FORBIDDEN` strings from Task 3, plus `_consensus_winner`, `mask_comparator`, `output_checker`, `run_reasonseg_best_flow`, and imports of deleted files. Exclude `docs/` and `tests/` because they intentionally name the boundary. Confirm both deleted modules are absent.

Expected: no matches and both file-existence checks return false.

- [ ] **Step 4: Audit the final API and README contract**

Confirm `process_image` has one return type and both branches return candidate lists; Reason candidate IDs are ordered Text Top1..N then FullDes; Refer retains nested raw metadata; CLI paths match README; and all 15 package versions exactly match the spec.

- [ ] **Step 5: Inspect the final change set**

Run: `git status --short` and `git diff --stat HEAD~4..HEAD` (or the equivalent range covering the implementation commits).

Expected: only planned product, test, config, and documentation files changed; no model weights, outputs, caches, IDE files, or unrelated formatting are included.

- [ ] **Step 6: Commit any verification-only corrections**

If Step 1-5 required corrections, commit only those corrections:

```bash
git add <exact corrected files>
git commit -m "fix: complete candidate-only release boundary"
```

If no corrections were required, do not create an empty commit.
