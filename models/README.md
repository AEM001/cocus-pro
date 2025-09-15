Qwen2.5-VL-7B-Instruct integration notes
=======================================

This repo includes a model-agnostic VLM interface and a stub integration for Qwen2.5-VL-7B-Instruct.

Options to connect a local Qwen VLM:

1) Local HTTP server (recommended)
- Run your own local server that accepts a JSON payload with fields:
  - `system`: system prompt string
  - `user`: user prompt string
  - `images`: list of `data:image/png;base64,...` encoded images (first one is used)
- Return a JSON object with a `text` field that contains the raw VLM response (JSON per prompts).
- Point the runner to it via `--server_url` or env `QWEN_VL_SERVER`.

2) In-process loading (advanced)
- Implement local loading in `src/sculptor/vlm/qwen.py:QwenVLM._infer_via_local` using your preferred runtime (Transformers, vLLM, etc.).
- Provide the model directory via `--model_dir` or env `QWEN_VL_MODEL_DIR`.

Directory convention
--------------------

Place the model under a directory of your choice, e.g.:

- /path/to/models/Qwen2.5-VL-7B-Instruct/

Then pass `--model_dir /path/to/models/Qwen2.5-VL-7B-Instruct` to the CLI.

SAM checkpoints
---------------

Place your SAM checkpoint under `models/`, e.g.:

- `models/sam_vit_h_4b8939.pth`

Run with:

```
python scripts/run_sculpt.py \
  --image ... --mask ... --roi_json ... --instance ... \
  --sam_checkpoint models/sam_vit_h_4b8939.pth --sam_type vit_h --sam_device cuda
```

Security & determinism
----------------------
- The current integration does not perform any network calls by default. The server call is a no-op stub returning `{}` until you implement it.
- The `MockVLM` can be used to validate the end-to-end pipeline without a real model.
