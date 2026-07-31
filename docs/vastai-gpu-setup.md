# GPU server setup playbook for tfmlens experiments

From zero to running one backbone's decoder finetune + self-repair sweep.
Every gotcha below was learned the hard way.

## 0. Machine
- vast.ai, official PyTorch image (ships CUDA torch under `/venv/main`)
- Reference spec: RTX 4090 / 24GB VRAM / 62GB RAM (cgroup limit ~64.6GB) / 16 CPU
- Run every long job inside tmux.

## 1. Connect to the box
SSH in using the private key whose public half is registered on the vast
instance. In this setup that key is **`arena_key`**, but **the name depends on
your setup** — if the connection fails, ask the user which key to use:

```bash
ssh -i ~/.ssh/arena_key <user>@<host> -p <port>
# or add it to ~/.ssh/config so plain `ssh <host>` works:
#   Host <host>
#     User <user>
#     Port <port>
#     IdentityFile ~/.ssh/arena_key
```

The same key is used for `scp` when copying results back (§5): pass `-i
~/.ssh/arena_key`.

## 2. Install deps — key trap: never `uv sync` on a GPU box
tfmlens's `pyproject.toml` pins torch to the CPU wheel index
(`[tool.uv.sources] torch = pytorch-cpu`, to keep local/CI light). So `uv sync`
overwrites CUDA torch with the CPU build and the GPU sits idle.

Correct path: use the image's preinstalled CUDA torch + plain pip (pip ignores
`[tool.uv.sources]`; torch is already satisfied so it won't reinstall):

```bash
git clone <repo> && cd tfmlens
git checkout <branch>                       # a fresh clone is on main — see note below
/venv/main/bin/pip install -e . tabicl     # core + tabicl group (synthetic prior)
# verify torch is still the CUDA build:
/venv/main/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect a +cuXXX version and True
```

Notes:
- **Branch matters — a fresh clone defaults to `main`.** Features under active
  development live on their branch, not `main`: e.g. the GT-logit sweep metric +
  `plot_self_repair.py --metric gt_logit` are on `feat/logit-metric`. Running the
  sweep from `main` silently gives you only the AUC trajectory. `git checkout` the
  branch that has the code you actually want **before** installing / running.
- Finetune only needs core + `tabicl` (synthetic prior); it does **not** need the
  `eval` group.
- For a different backbone, swap `tabicl` for the matching dep group:
  - mitra: `einx safetensors huggingface-hub` (group `mitra`)
  - tabfm: `absl-py safetensors huggingface-hub` (group `tabfm`)

## 2b. OpenML offline cache (needed for the sweep, not for finetune)
openml.org has had a global 504, and the server usually can't reach out anyway.
openml can read **offline from cache**, but the path must be right:
- Server default: `~/.cache/openml/org/openml/www` (note: **not** `~/.openml`)
- From your local box, push `~/.openml/org/openml/www/{tasks,datasets}` up to that
  path — but use the robust transfer below, **not** a plain scp/rsync.

**Trap: "false success" on a flaky link.** A plain `rsync`/`scp` can exit **0**
while the connection dropped mid-transfer (vast "session limit" kills long
connections) — the exit code lies, the dir lands **empty or truncated**. Use
`--partial` + keepalive + a retry loop, and only trust the transfer once the
loop's own "done" marker prints (see §5 for the same pattern on the way back).

## 3. Run the finetune (train per-layer decoders)

**On a fresh / recreated box you usually don't need to re-finetune.** The sweep
(§4) only needs the trained decoders in `weights/<model>/` (N `.pth` files). If
you already have them locally (from a prior run), upload that dir with the robust
transfer (§5) and **skip straight to §4** — finetuning is ~23min/model of wasted
GPU. Re-finetune only when the decoders don't exist yet.

Bootstrap order on a brand-new instance: §1 connect → §2 clone + right branch +
deps → upload `weights/<model>/` **and** the openml cache (§2b) → §4 sweep.

To actually train, inside tmux:

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
Inside tmux, and **`mkdir -p out` first** (see the redirect trap below):
```bash
mkdir -p out
tmux new -d -s sweep "/venv/main/bin/python scripts/run_self_repair_sweep.py \
  --model <model> --subsample-train 1000 --subsample-test 500 \
  --skip-diffs --device cuda --out out/<model>.json > out/<model>.log 2>&1"
```

- **`--subsample-train 1000 --subsample-test 500` is not arbitrary** — it matches
  the paper's setup and the known-good reference runs. Going smaller (e.g.
  300/100) is *not* just faster-and-noisier: fewer in-context rows make the model
  weaker **and** make each ablation bite harder, so the self-repair curves shift.
  Keep 1000/500 unless you have a reason not to.
- **tmux redirect chicken-and-egg:** `tmux new -d '... > out/x.log'` dies
  **silently** ("no server running") if `out/` doesn't exist yet — the shell
  can't open the redirect target, so the session never starts. Always
  `mkdir -p out` *before* launching tmux, not inside it.
- High-feature tables may OOM on VRAM: rerun those alone with
  `--tasks <ids> --subsample-train 3000`, then merge the json.
- Plot: `python scripts/plot_self_repair.py --model <model>`

## 5. Copy files over a flaky / session-limited link (both directions)
Same root cause as §2's "false success": the vast box drops long connections, so
scp/rsync can die mid-transfer yet exit **0** or leave a truncated file. This
bites both the openml upload (§2) and pulling `weights/<model>/` + `out/*.json`
back.

Robust pattern — `--partial` (keep half-sent files, resume) + `ServerAliveInterval`
(keepalive, fewer drops) + `until … do sleep; done` (auto-retry until it *really*
finishes; only then print the marker):

```bash
SSHOPT="ssh -q -i ~/.ssh/arena_key -p <port> -o ServerAliveInterval=15 -o ServerAliveCountMax=3"
until rsync -a --partial -e "$SSHOPT" root@<host>:tfmlens/out/<file>.json out/; do
  echo "[retry] $(date)"; sleep 10
done
echo "DONE"          # trust the transfer only after this prints
```

- Do the whole thing inside a detached background script for big dirs (weights
  can be ~800MB) so a dropped **local** shell doesn't abort the retry loop.
- Final verify (optional): `ls -l` a size manifest on both ends and `diff` them.

## 6. vast / ssh misc gotchas
- `pkill -f <pattern>` also kills the ssh command itself (the pattern matches its
  own command line); stop a job with `tmux kill-session -t <name>`.
- Don't chain kill+pull+launch into one giant ssh command — it tends to drop
  mid-way with "no output"; split into several short commands.
- Keep long jobs in tmux so a dropped connection doesn't matter.
