# GPU 服务器上跑 tfmlens 实验的 setup playbook

从零到能跑一个 backbone 的 finetune + self-repair sweep。踩过的坑都标出来了。

## 0. 机器
- vast.ai,PyTorch 官方镜像(自带 CUDA torch,装在 `/venv/main`)
- 参考规格:RTX 4090 / 24GB VRAM / 62GB RAM(cgroup 上限约 64.6GB)/ 16 CPU
- 长任务全部丢 tmux 里跑

## 1. 装依赖 —— 关键坑:GPU 机器上绝对不要 `uv sync`
tfmlens 的 `pyproject.toml` 把 torch pin 到了 CPU wheel index
(`[tool.uv.sources] torch = pytorch-cpu`,为了让本地/CI 轻量)。
所以 `uv sync` 会把 CUDA torch 覆盖成 CPU 版,GPU 直接闲置。

正确做法:用镜像自带的 CUDA torch + 普通 pip(pip 忽略 `[tool.uv.sources]`,
torch 已满足就不会重装):

```bash
git clone <repo> && cd tfmlens
/venv/main/bin/pip install -e . tabicl     # core + tabicl 组(合成 prior 用)
# 验证 torch 还是 CUDA 版:
/venv/main/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望输出带 +cuXXX 且 True
```

说明:
- finetune 只需要 core + `tabicl`(合成 prior),**不需要 `eval` 组**。
- 换别的 backbone 就把 `tabicl` 换成对应 dep 组:
  - mitra: `einx safetensors huggingface-hub`(组名 `mitra`)
  - tabfm: `absl-py safetensors huggingface-hub`(组名 `tabfm`)

## 2. OpenML 离线缓存(sweep 阶段需要,finetune 阶段不需要)
openml.org 有过全局 504,而且服务器上一般连不出去。openml 支持**从缓存离线读**,
但缓存路径必须对:
- 服务器默认路径:`~/.cache/openml/org/openml/www`(注意不是 `~/.openml`)
- 从本地机器把 `~/.openml/org/openml/www/{tasks,datasets}` scp 到上面那个路径

## 3. 跑 finetune(训练 per-layer decoders)
在 tmux 里:

```bash
tmux new -s ft
cd tfmlens && git pull
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /venv/main/bin/python -m tfm_lens.finetune \
  --config configs/<model>.yaml --model <model>
```

### 24GB 卡上的 config 坑(以 `configs/tabfm.yaml` 为例,写死在 yaml 里)
被 OOM 反复教育出来的,换模型也大同小异:

- **`prior.max_seq_len: 512`(不要 1024)**
  行注意力是 O(seq²)。1024 → 每步几分钟(全跑 ~16h);512 → 每步 14–70s。
  decoder 的可解码性不需要满 1024 上下文。

- **`prior.n_jobs: 1`(不要 4/16)**
  loky 预取 worker + 模型的 CUDA context 会互撞,导致 EXIT_1 / 137。
  串行 prior 生成没关系,瓶颈本来就是 forward。

- **`prior_batch_size: 128`(不要默认 512)—— 最主要的 OOM 源**
  每一步会把「所有 depth × prior_batch_size 张表的 test-row 读出」全部堆在
  **host RAM** 里(`finetune_decoders.py:68-85`,torch.cat 还会瞬时翻倍)。
  512 → 撞 64.6GB cgroup 上限 → 第 2 步被 SIGKILL(退出码 137,**没有 CUDA
  traceback,因为爆的是主机内存不是显存**)。128 → 活跃 RAM ~27GB。

- `micro_batch_size: 8`,`max_steps: 100`(200 是 overkill,100 就收敛),
  `save_every: 50`(长跑中途存盘,防止卡死全丢)。
- 参考耗时:tabfm 全量 100 步 finetune ~23min,forward 峰值显存 ~10GB。

判断退出码:
- **137** = 主机 RAM 撞 cgroup 上限(降 `prior_batch_size`)
- **EXIT_1** = 多半是 loky worker + CUDA(设 `n_jobs: 1`)
- OOM 但有 CUDA traceback = 显存(降 `micro_batch_size` 或 `max_seq_len`)

## 4. 跑 self-repair sweep(出 Figure 8 数据)
```bash
/venv/main/bin/python scripts/run_self_repair_sweep.py \
  --model <model> --subsample-train 1000 --subsample-test 500 \
  --skip-diffs --device cuda
```

- 大特征数的表可能显存 OOM:单独 `--tasks <ids> --subsample-train 3000` 重跑,
  再把 json 合并即可。
- 出图:`python scripts/plot_self_repair.py --model <model>`

## 5. 把结果拷回本地
- 权重在 `weights/<model>/`(N 个 decoder `.pth`),结果 json 在 `out/`。
- scp 有 300s 超时,大目录容易只传一半、还可能有**截断文件**(存在但不完整,
  不会出现在「缺失」列表里)。稳妥做法:两边各 `ls -l` 出文件大小清单,
  `diff` 对比,专门重传大小对不上的那几个,最后确认所有文件字节数一致。

## 6. vast / ssh 杂项坑
- `pkill -f <pattern>` 会连 ssh 命令自己一起杀(pattern 匹配到自身命令行);
  停任务用 `tmux kill-session -t <name>`。
- ssh 里别把 kill+pull+launch 串成一条超长命令,容易中途断连、「无输出」;
  拆成几条短命令分别执行。
- 长任务永远在 tmux 里,断线不影响。
