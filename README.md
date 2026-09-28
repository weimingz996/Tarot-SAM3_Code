# Tarot-SAM3

Tarot-SAM3 is a training-free framework for explicit and implicit referring
expression segmentation with SAM3.

## Public release boundary

This repository exposes the pipeline only through ERI candidate generation. It
stops before the final best-mask vote and contains no Mask Self-Refinement (MSR)
implementation. Consequently, the public code produces multiple candidate masks
and does not produce a final segmentation prediction.

- Reason mode returns the Text Top3 candidates followed by the FullDes candidate.
- Refer mode returns the pre-consensus `masks_info`, including the selected
  representative bbox candidate when one is available.

## Python API

`TarotSAM3.process_image(...)` returns `list[dict]`. Every candidate record keeps
its metadata and includes at least:

- `id`: stable candidate label;
- `source`: candidate origin;
- `mask`: a boolean NumPy mask.

Empty masks are preserved because they are part of the candidate-generation
record. Callers that need one final mask must implement their own downstream
selection outside this repository.

## Command-line output

```bash
python tarot_sam3.py \
  --config configs/7B.yaml \
  --image_path path/to/image.jpg \
  --query "the referred object" \
  --save_dir output
```

Add `--reason_seg` for Reason mode. The command writes every candidate to:

```text
output/candidate_masks/01_<candidate-id>.png
output/candidate_masks/02_<candidate-id>.png
...
```

Candidate IDs are lowercased and sanitized for filenames. The numeric prefix
preserves order and uniqueness. No `final_output_mask.png` is created.

## SAM3 dependency

This repository does not vendor the third-party SAM3 implementation, model
weights, or tokenizer vocabulary. Install the official SAM3 source package so
that `sam3.model_builder` and `sam3.model.sam3_image_processor` are importable.
The installed package must include `sam3/assets/bpe_simple_vocab_16e6.txt.gz`.
Set `sam3.weight_path` in `configs/7B.yaml` to the downloaded checkpoint.

## Reference environment

The versions below were read from the server environment at
`/hpc2hdd/home/wzhang915/.conda/envs/SAM123` (Python 3.11.14).

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
