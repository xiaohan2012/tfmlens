# 参考实现:Exp4 训练 → Exp6 推理 → 出图(已在 2026-07-23 端到端跑通)

> **用途**:这是 tfm_lens 重写的**行为契约**。旧代码库(`is_one_layer_enough`)在 LimiX-2M 上把这条链路完整跑通过一次,本文把「实际发生了什么、产出什么、每步的输入输出契约、踩过的坑」固化下来。重写时,新库的 `training/` + `experiments/` + `plotting/` 要复刻这里的**行为**(而非旧代码的实现方式)。
>
> 配套:`DESIGN.md`(新架构)、`docs/old-code-exp4-walkthrough.md`(旧训练代码白盒)、`docs/paper/`(论文)。原始复现记录见 ml-reading `notes/exp4-exp6-repro-plan.md` §九。

---

## 0. 三段流水线全景

```
[Stage 1] 训 per-layer decoder (Exp4)  ── 唯一花算力,产出可复用资产
    ↓ 13 个 decoder_layer_*.pth
[Stage 2] 推理:正常前向(c0)+ 跳层前向(c2) 逐层解码 (Exp6/Exp5)  ── 纯推理,便宜
    ↓ per-task 结果 pkl(含 decoder / finetuned_decoder 两组 probing)
[Stage 3] 出图:读 pkl → 归一化 → 逐层曲线 + self-repair 重连线
    ↓ self-repair PDF
```

**已验证数字(LimiX-2M · Vast RTX 4090)**:Stage 1 ≈ 5.3h;Stage 2 ≈ 7s/跳层配置(单任务 13 配置 ≈ 90s 计算 + 一次性 OpenML 下载);Stage 3 ≈ 几秒。

---

## 1. Stage 1 — 训 per-layer decoder(Exp4)

**命令**(旧代码):
```
WANDB_MODE=offline HF_HUB_DISABLE_XET=1 \
python -u fine_tuning_exp.py --config limix_2m.config_finetuning_decoder
```

**已验证配方**(= 论文 200 epoch × 512 步/epoch):
- 冻结主干,**每层一个 decoder**:`copy.deepcopy(model.decoder)` × (n_layers+1)。LimiX-2M 12 层 → **13 个 decoder**。
- 每个 decoder 独立 `AdamW(lr=3e-5, weight_decay=1e-4)`,`clip_grad_norm=1.0`,`CrossEntropyLoss`。
- 训练量:`training_max_steps=200`,每步 `prior_batch_size=512` 张合成表 → 102,400 表。`micro_batch_size=64`(512/64=8 微批),`training_batch_size=8`(= 论文 batch)。
- **数据 = on-the-fly**:`PriorDataset(prior_type="mix_scm", ...)`,不落盘。参数(核对论文附录 `tabicl_prior_config`):`batch_size_per_gp=4, min/max_features=2/30, max_classes=10, max_seq_len=1024, log_seq_len=False, min/max_train_size=0.1/0.9, n_jobs=-1`。⚠️ **`prior_type` 必须 `mix_scm`**(70% mlp_scm + 30% tree_scm);CLI 默认 `graph_scm` 是坏值。
- **只训 test 段**:loss 只对 `emb[:, train_size:]`(test 行)算。
- **`separate_y_embeddings` 语义(影响速度极大)**:
  - `True`(LimiX 系、TabPFN v2/v2.5):decoder 只吃 label token `emb[:, :, -1, :]` → 快。
  - `False`(TabPFN v1、TabICL):decoder 吃整条序列 → 每步慢 ~3×(实测 LimiX-2M 95s/步 vs TabPFN v1 258s/步)。

**产出**(落 `weights/extras/<model>/`):`decoder_layer_0..N.pth` + `loss_per_step.pkl`。**这是 Stage 2/3 复用的唯一资产,不可再生(丢了重训 5h)。**

**关键实现支点**:每层表示靠**前向副作用**抓取(旧代码在 layer forward 里塞 `self.out_embeddings`)。→ **新库改用 forward hook**,见 DESIGN.md §3。

**运维坑**:单 GPU 不能并行(kernel-launch 排队 ~3× 慢);loky worker 不能 `pkill -9`(污染 multiprocessing resource tracker)。

---

## 2. Stage 2 — 推理(Exp6 self-repair / Exp5 looping)

**这一步不再训练**,加载 Stage 1 的 decoder,在真实数据集上跑冻结前向 + 逐层解码。

**命令**(旧代码,`--lite_evaluation` = 单 repeat 单 fold 快跑):
```
python main.py --benchmark TabArena --task_id <tid> --config limix_2m.config_c0 --lite_evaluation
python main.py --benchmark TabArena --task_id <tid> --config limix_2m.config_c2 --lite_evaluation
```
- **c0** = 正常前向(无干预)→ 基线逐层轨迹。
- **c2** = 跳掉第 i 层,一个 config 自动展开成 **12 个 skip 变体**(p0..p11)。
- Exp5(looping)= 重复/循环某层的干预,同理换一组 config。

**必需的 config 改动**(⚠️ 旧代码默认是坏的):`config_c0.py` / `config_c2.py` **删掉 `pop("finetuned_decoders_path")`** → 触发 `main.py collect_finetuned_decoder_results` → 同时产出 `decoder`(默认头)+ `finetuned_decoder`(我们训的头)两组。不删就画不出逐层 logit lens。

**产出契约**(每 task 一个 pkl,`results/<benchmark>/<tid>/<repeat>_<fold>/<model>_<config>.pkl`):
```
{ task_id, task_type, n_classes, config, y_test, y_pred, y_pred_proba,
  test_scores,                       # 全模型基线分 → 映射为 layer="main", probing="decoder"
  layer_0, layer_1, ..., layer_N {   # 每层
     test_scores,                    # 默认 decoder 逐层 → probing="decoder"
     finetuned_decoder { test_scores, train_scores, y_pred_proba },  # 我们的 decoder → probing="finetuned_decoder"
     embedding_distances, ...        # c0 才有(embedding_similarity=True)
  } }
```

**性能画像**:每个 skip 配置 ~7s;单任务瓶颈是 **c0 的 `embedding_distances`**(O(n²) CPU,~2-4 分钟),不是推理。→ 新库出 self-repair 图**不需要** embedding_distances,可关掉省时间。

---

## 3. Stage 3 — 出图(self-repair)

**脚本**:旧 `plots/04a_skipping_layer.py::plot_skipping_finetuned`;我们的裁剪版 `plot_selfrepair_limix.py`(已备份到本地 repo)。

**数据契约**(plot 读什么):
- 基线全模型分:`base_config=="c0"`, `layer=="main"`, `probing=="decoder"`。
- 逐层 logit-lens 轨迹:`base_config=="c0"`, `layer!="main"`, `probing=="finetuned_decoder"`。
- 跳层轨迹:`base_config=="c2"`, `probing=="finetuned_decoder"`。
- 每个数据集先按自己的 c0-main 分**归一化**,再跨数据集取均值(`estimator="mean"`)。

**两个必踩的坑**:
1. `load_results`(exp_utils.py:308)任一 model/config **缺 pkl → 整个 task 进 `task_id_with_errors` → 从图里删掉**。所以只画一个模型时,**必须把 `model_configs` 裁到只剩该模型**,否则数据被没结果的其他模型清空。
2. `--lite_evaluation` 只产 `0_0` 单折,但 `load_results` 默认按硬编码 `repeats×folds` 找文件 → 缺文件报错删任务。**出图前强制 `repeats=folds=1`**。

**输出**:`c2_skipping_layer_{all,TabArena}.pdf`。图见 ml-reading `notes/figures/balef2026/repro/selfrepair_limix2m_{1task,4tasks}.png`,结构与论文 LIMIX-2M 面板一致。

---

## 4. 映射到 tfm_lens 模块(重写时的对应关系)

| 本文 Stage | 旧代码 | → tfm_lens 模块(DESIGN.md) | 重写要点 |
|---|---|---|---|
| Stage 1 训练 | `fine_tuning_exp.py` | `training/train_decoders.py` + `data/prior.py` | 用 hook 抓表示取代 out_embeddings;prior 参数集中一处;`separate_y_embeddings` 收进 adapter 能力位 |
| Stage 2 推理 | `main.py` + `config_c0/c2` | `experiments/exp6_self_repair.py` + `core/interventions.py` + `core/logit_lens.py` | skip 干预用 context manager(取代 task_info 权重穿透 + 删 config 键);probing 两组结果由 logit_lens 统一产出 |
| Stage 3 出图 | `plots/04a_*.py` | `plotting/` | 数据契约保持;但设计上避免「缺一个模型清空全部」这种脆弱耦合 |

**一句话**:新库要让本文列出的每个「⚠️ 坑」在**设计层面不可能发生**(干预是显式 context manager 而非删 config 键;抓表示是 hook 而非源码副作用;prior_type/参数有单一可信来源;绘图数据按模型独立不互相连累)。

---

## 5. 已验证资产位置(本地,box 已可关)
- decoder:`code/is_one_layer_enough/FoundationModels/weights/extras/limix_2m/`(13 pth + loss_per_step.pkl)
- 基座:`.../weights/Limix/LimiX-2M.ckpt`
- 推理结果:`.../Experiments/results/TabArena/{363619,363621,363623,363626}/0_0/`
- 出图脚本:`.../Experiments/plots/plot_selfrepair_limix.py`
- 图:ml-reading `notes/figures/balef2026/repro/`
