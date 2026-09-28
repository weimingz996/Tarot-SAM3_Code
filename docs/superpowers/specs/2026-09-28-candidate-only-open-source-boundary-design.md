# Candidate-Only Open-Source Boundary

## Goal

Publish Tarot-SAM3 only through ERI candidate generation. The public pipeline
must return the masks that would have entered final best-mask selection, while
the repository must not contain the final voting or downstream mask
self-refinement (MSR) implementation.

The release must support both ReasonSeg and referring-expression inputs. It
must not claim that any returned mask is a final segmentation result.

## Non-goals

- Do not replace the removed vote with a simpler heuristic.
- Do not select a preferred candidate in the CLI or Python API.
- Do not redesign candidate generation.
- Do not remove internal bbox filtering that is needed to construct the Refer
  candidate set.
- Do not add new dependencies or configuration layers.

## Public output contract

`TarotSAM3.process_image(...)` returns `list[dict]` instead of one NumPy mask.
Each record contains the candidate's existing metadata and a two-dimensional
boolean NumPy array under `mask`. At minimum, records expose an identifier,
source, and mask. Existing metadata is preserved when available.

The list is the exact top-level input that would have reached final
best-mask selection. Empty candidate masks remain in the list so the boundary
does not silently alter candidate generation.

The CLI writes each top-level candidate to:

```text
<save_dir>/candidate_masks/01_<candidate-name>.png
<save_dir>/candidate_masks/02_<candidate-name>.png
...
```

Names are deterministically sanitized for filenames. The existing experiment
logger records each candidate's identifier, source, output path, and foreground
pixel count. The CLI no longer creates `final_output_mask.png`.

If generation yields no candidates, `process_image` returns an empty list and
the logger records that condition. It does not synthesize an empty final mask.

## Reason candidate flow

The Reason path keeps:

1. Full-description generation.
2. Text augmentation.
3. SAM text-mask generation and Text Top3 construction.
4. FullDes mask generation.
5. Candidate archive and metadata creation needed by the public candidate
   stage.

It returns the selectable Text Top3 and FullDes records before
`select_best_mask(...)`. Bbox candidates remain generation evidence but are not
returned as selectable outputs because the existing final vote does not allow
them to become the output mask.

The Short/Long branch and A3.x vote/certificate stages occur after an incumbent
has already been selected, so they are removed from the public flow and source
tree.

The public runner is renamed from a best-flow concept to a candidate-flow
concept. Historical filenames may remain where renaming would create a large,
content-neutral diff, but public symbols and docstrings must use candidate-only
terminology.

## Refer candidate flow

The Refer path keeps its current description generation, SAM text masks, bbox
mask generation, bbox filtering, and selection of the bbox representative used
to form the final candidate list.

It stops immediately before `_consensus_winner(masks_info)` and returns
`masks_info`. Nested `raw_masks_info` remains attached to its parent record.
This preserves the exact candidate structure that the removed final vote would
have received.

The earlier bbox representative selection remains because it constructs one
of the final vote candidates; it is not the cross-candidate best-mask vote that
defines the open-source boundary.

## Source removal

Delete or strip the following implementation:

- The Reason Short/Long voting module
  `reason_prompt/reasonseg_shortlong_bbox_experiment.py`.
- Final Reason vote construction, certificates, refinements, and
  `select_best_mask` from `reasonseg_best_flow_reference.py`.
- The unused Refer final selector module
  `ref_prompt/mask_candidate_selector.py`.
- `_consensus_winner` and every downstream MSR method in `tarot_sam3.py`.
- MSR-only prompts from `src/prompts.py`.
- DINO/MSR-only model and utility code whose repository-wide callers disappear
  after the main-flow removal.
- Imports, state, configuration fields, and logging that become unused because
  of these deletions.

Candidate-generation helpers, generic mask operations still used before the
boundary, and `ref_prompt/training_free_bbox_selector.py` remain.

## Configuration

Remove configuration that only controls deleted code, including the DINOv3
section, the MSR section, Reason final-vote parameters, and Reason/Refer point
sampling settings used only by MSR.

Keep configuration used before the boundary, including SAM confidence,
description-generation controls, bbox filtering, Refer variant generation, and
the Refer exact-copy guard.

## README and environment record

Update the README to:

- Describe the repository as a candidate-generation release.
- State clearly that it does not select or return a final mask.
- Document the Python API return type and CLI candidate-mask directory.
- Record that versions were captured from the server's `SAM123` Conda
  environment at `/hpc2hdd/home/wzhang915/.conda/envs/SAM123`.
- List Python 3.11.14 and these 15 core package versions:

| Package | Version |
| --- | --- |
| torch | 2.8.0 |
| torchvision | 0.23.0 |
| transformers | 4.57.0 |
| accelerate | 1.11.0 |
| numpy | 1.26.4 |
| opencv-python | 4.10.0.84 |
| Pillow | 10.4.0 |
| scipy | 1.16.3 |
| scikit-learn | 1.3.2 |
| matplotlib | 3.10.7 |
| seaborn | 0.13.2 |
| PyYAML | 6.0.2 |
| requests | 2.32.3 |
| tqdm | 4.66.5 |
| triton | 3.4.0 |

This is an environment snapshot, not a promise that every package remains a
direct dependency after candidate-only pruning.

## Error handling

Existing input, model, and candidate-generation errors continue to propagate.
No fallback may call or recreate deleted voting behavior. An empty candidate
set is a valid candidate-generation result and is logged explicitly.

## Verification

Implementation follows a minimal red-green workflow:

1. Add a focused test that expects Reason and Refer paths to return candidate
   records and proves final vote/MSR callbacks are not reachable.
2. Change the pipeline until that test passes.
3. Verify candidate PNG naming and empty-list behavior.
4. Run the available test suite and Python compilation checks.
5. Scan the repository for removed public symbols, MSR entry points, final-mask
   output names, and imports of deleted modules.
6. Confirm README versions exactly match the captured server environment.

Completion requires no final-vote or MSR implementation to remain in the
public source tree, while both public modes still produce their pre-vote
candidate records.
