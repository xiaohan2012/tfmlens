# GPU server setup playbook for tfmlens experiments

From zero to running one backbone's decoder finetune + self-repair sweep.
Every gotcha below was learned the hard way.

## 0. Machine
- vast.ai, official PyTorch image (ships CUDA torch under `/venv/main`)
- Reference spec: RTX 4090 / 24GB VRAM / 62GB RAM (cgroup limit ~64.6GB) / 16 CPU
- Run every long job inside tmux.

## 1. Install deps — key trap: never `uv sync` on a GPU box
tfmlens's `pyproject.toml` pins torch to the CPU wheel index
(`[tool.uv.sources] torch = pytorch-cpu`, to keep local/CI light). So `uv sync`
overwrites CUDA torch with the CPU build and the GPU sits idle.

Correct path: use the image's preinstalled CUDA torch + plain pip (pip ignores
`[tool.uv.sources]`; torch is already satisfied so it won't reinstall):

```bash
git clone <repo> && cd tfmlens
/venv/main/bin/pip install -e . tabicl     # core + tabicl group (synthetic prior)
# verify torch is still the CUDA build:
/venv/main/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect a +cuXXX version and True
```

Notes:
- Finetune only needs core + `tabicl` (synthetic prior); it does **not** need the
  `eval` group.
- For a different backbone, swap `tabicl` for the matching dep group:
  - mitra: `einx safetensors huggingface-hub` (group `mitra`)
  - tabfm: `absl-py safetensors huggingface-hub` (group `tabfm`)

## 2. OpenML offline cache (needed for the sweep, not for finetune)
openml.org has had a global 504, and the server usually can't reach out anyway.
openml can read **offline from cache**, but the path must be right:
- Server default: `~/.cache/openml/org/openml/www` (note: **not** `~/.openml`)
- From your local box, scp `~/.openml/org/openml/www/{tasks,datasets}` up to that
  path.

## 3. Run the finetune (train per-layer decoders)
Inside tmux:

```bash
tmux new -s ft
cd tfmlens && git pull
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /venv/main/bin/python -m tfm_lens.finetune \
  --config configs/<model>.yaml --model <model>
```

### 24GB-card config gotchas (example `configs/tabfm.yaml`, baked into the yaml)
Learned from repeated OOMs; similar across models:

- **`prior.max_seq_len: 512` (not 1024)**
  Row attention is O(seq²). 1024 → minutes/step (~16h total); 512 → 14–70s/step.
  Decoder decodability doesn't need the full 1024 context.

- **`prior.n_jobs: 1` (not 4/16)**
  loky prefetch workers + the model's CUDA context crash each other → EXIT_1 / 137.
  Serial prior generation is fine; the forward is the bottleneck anyway.

- **`prior_batch_size: 128` (not the default 512) — the main OOM source**
  Each step parks all depths' test-row readouts for `prior_batch_size` tables in
  **host RAM** (`finetune_decoders.py:68-85`; torch.cat also doubles it
  transiently). 512 → hits the 64.6GB cgroup limit → step 2 gets SIGKILL (exit
  137, **no CUDA traceback because it's host memory, not VRAM**). 128 → live RAM
  ~27GB.

- `micro_batch_size: 8`, `max_steps: 100` (200 is overkill; 100 converges),
  `save_every: 50` (checkpoint mid-run so a stall doesn't lose everything).
- Reference timing: tabfm full 100-step finetune ~23min; forward peak VRAM ~10GB.

Exit-code triage:
- **137** = host RAM hit the cgroup limit (lower `prior_batch_size`)
- **EXIT_1** = usually loky workers + CUDA (set `n_jobs: 1`)
- OOM with a CUDA traceback = VRAM (lower `micro_batch_size` or `max_seq_len`)

## 4. Run the self-repair sweep (Figure 8 data)
```bash
/venv/main/bin/python scripts/run_self_repair_sweep.py \
  --model <model> --subsample-train 1000 --subsample-test 500 \
  --skip-diffs --device cuda
```

- High-feature tables may OOM on VRAM: rerun those alone with
  `--tasks <ids> --subsample-train 3000`, then merge the json.
- Plot: `python scripts/plot_self_repair.py --model <model>`

## 5. Copy results back to local
- Weights in `weights/<model>/` (N decoder `.pth` files), result json in `out/`.
- scp has a 300s timeout; large dirs often transfer half-way and can leave a
  **truncated file** (exists but incomplete — it won't show up in the "missing"
  list). Safe approach: `ls -l` a size manifest on both ends, `diff` them,
  re-copy only the mismatches, then confirm every file's byte count matches.

## 6. vast / ssh misc gotchas
- `pkill -f <pattern>` also kills the ssh command itself (the pattern matches its
  own command line); stop a job with `tmux kill-session -t <name>`.
- Don't chain kill+pull+launch into one giant ssh command — it tends to drop
  mid-way with "no output"; split into several short commands.
- Keep long jobs in tmux so a dropped connection doesn't matter.
