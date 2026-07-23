# Exp4(Tabular Logit Lens / fine-tune decoder)代码走读与评价

> 对应论文:Balef et al. *Is One Layer Enough?* (ICML 2026),第 4 个实验。
> 代码库:`code/is_one_layer_enough/`。本文只覆盖 Exp4 这条主线(训练 per-layer decoder),它是全文的核心;Exp6(self-repair)直接复用 Exp4 训好的 decoder,不再训练。

---

## 0. 一句话概括

Exp4 只干一件事:**给每一层 transformer 各训练一个独立的 decoder 头**,这样就能把「第 L 层的中间表示」直接翻译成类别预测 —— 这就是论文说的 **tabular logit lens**。

- 骨干网络(LimiX-2M / TabPFN 等)**全程冻结**,只训 decoder。
- 预训练模型本来只有**一个** decoder(挂在最后一层)。Exp4 把它 `deepcopy` 成 `n_layers+1` 份,每份单独训练,于是能在**任意深度**读出预测。
- Exp6 加载这些 `decoder_layer_{L}.pth`,在「跳过某一层」的干预下再读一次,衡量 self-repair —— **不再训练**。

---

## 1. 代码流程(从入口往下)

整条链路都在 `Experiments/fine_tuning_exp.py`,入口 `main()` @ `:343`。

### 1.1 配置 (`:347-349`)
- CLI:`--config limix_2m.config_finetuning_decoder` → dotted path 动态 import。
- config 三层叠加:
  - `configs/limix_2m/config_base.py` —— 骨干 ckpt 路径、`number_of_layers=12`、`layers_info`。
  - `configs/limix_2m/config_finetuning_decoder.py` —— 训练超参 + 我们加的 on-the-fly 段。

### 1.2 数据 `build_dataloader` (`:51`)
- `on_the_fly_prior=True` → `PriorDataset`(`tabicl.prior.dataset`),无限迭代器,每次吐一个 `batch=512` 张合成表。
- `False` → `LoadPriorDataset`,从磁盘读预生成文件(原始行为,我们不用)。
- 每个样本是 5-tuple:`(X, y, d, seq_len, train_size)`。
- prior 参数(见 config,已核对论文附录):`mix_scm` = 70% MLP-SCM + 30% tree-SCM;`min/max_features=2/30`;`max_classes=10`;`max_seq_len=1024`;`train_size` 占比 0.1~0.9。CPU 多核生成(`n_jobs=-1`)。

### 1.3 模型 + decoders (`:367-369`)
- `clf_class(**model_parameters, random_state=0)` → 加载**冻结**的 LimiX-2M 骨干。
- `build_decoders_and_optimizers` (`:103`):`copy.deepcopy(clf.get_decoder())` 共 `n_layers+1 = 13` 次,每份配独立 `AdamW(lr=3e-5, weight_decay=1e-4)`。

### 1.4 训练循环 (`:379-385`) —— 200 步,每步一个新 batch
每步进 `run_on_micro_batches` (`:217`),核心逻辑:
1. 512 张表切成 8 个 micro-batch(每个 `micro_batch_size=64`)。
2. 对每个 micro-batch:
   - `with torch.no_grad(): clf.model_inference(X, y_train, train_size)` (`:263-264`) —— **冻结前向**。
   - `_prepare_embeddings` (`:176`) 把每层表示掏出来 → `torch.stack` → `(13, batch, seq, hidden)`。
3. 梯度累积:攒够 `training_batch_size=8` 个样本就对**每一层各做一次** decoder 更新(`gradient_acc_steps = training_batch_size // micro_batch_size`)。
4. `train_step` (`:130`):只取 test 段 `emb[:, train_size:]` → decoder 前向 → CE loss vs test 标签 → backward → `clip_grad_norm_(1.0)` → step。

**数据量核对**:200 步 × 512 表 = 102,400 张表,对应论文「200 epochs × 512 steps/epoch」。tqdm 里的 `8` 是 micro-batch 个数,不是 batch size。

### 1.5 Checkpoint `save_checkpoint` (`:315`)
- 每 `save_every_step=100` 步存 `decoder_layer_{0..12}.pth` + `loss_per_step.pkl` 到 `FoundationModels/weights/extras/limix_2m/`。

---

## 2. 最关键、也最"取巧"的一处:embedding 靠前向副作用抓取

`FoundationModels/Limix/limix/sklearn/classifier.py:177-182`:

```python
def get_all_layers_embeddings(self):
    embeddings = []
    for l, info in self.layers_info:
        layer = self.predictor.model.transformer_encoder.layers[l]
        embeddings.append(layer.out_embeddings)   # ← 读取前向时缓存的副作用
    return embeddings
```

- `model_inference` 前向时,每个 layer 把自己的输出**偷偷存**到 `layer.out_embeddings`。
- 所以必须**先跑 `model_inference`,再读 embeddings**,顺序反了就是旧值。
- 这是整个 logit-lens 的实现支点,也是它最脆的地方 —— 隐式全局状态,没有任何显式返回。

`separate_y_embeddings=True` 时只留标签 token `emb[:, :, -1, :]`(`fine_tuning_exp.py:193`)—— 因为 LimiX/TabPFN 把 label 作为单独 token 拼进序列,预测读的就是这个 token 的表示。

推理侧对称实现见 `classifier.py:187 get_all_layers_predictions`(`decoder_type="finetuned"` 时用 `self.finetuned_decoders[l]`),以及 `:233 load_finetuned_decoders`(Exp6 加载 Exp4 权重的入口)。

---

## 3. 关键方法索引(各模型 wrapper)

| 方法 | 作用 | LimiX 位置 |
|---|---|---|
| `model_inference(X, y, support_size)` | 冻结前向,填充 `out_embeddings` | `classifier.py:243` |
| `get_all_layers_embeddings()` | 读每层缓存表示 | `classifier.py:177` |
| `get_decoder()` | 骨干自带的 decoder 头(被 deepcopy 的模板) | `classifier.py:229` |
| `get_decoder_pre_norm()` | decoder 前的 LayerNorm(`encoder_out_norm`) | `classifier.py:184` |
| `load_finetuned_decoders()` | Exp6 加载 `decoder_layer_{L}.pth` | `classifier.py:233` |

TabPFN_v1 / v2 / TabICL 有对称的同名方法(见 `grep -rn "def get_all_layers_embeddings"`)。

---

## 4. 对代码库的真实评价

分两层看,差距很大。

### 4.1 实验驱动脚本(`fine_tuning_exp.py`)—— 意外地干净 👍
- 有 docstring、类型标注、职责拆分清楚,像被人认真重构过。
- 仍有代码味:`_log_losses` 里 `steps==0` 那个分支(`training_batch_size < micro_batch_size` 时走 sub-batch 循环)是为兼容两种累积模式硬凑的,很绕。

### 4.2 周边基础设施 —— 典型学术研究代码,坑不少 👎
- **死配置**:两个 config 里 `prior_data_dir` 还指向一个不存在的目录,虽然 `on_the_fly=True` 根本不用它,但会误导人。
- **脆弱耦合**:base 里 set `finetuned_decoders_path`,finetuning config 里又 `del` 掉 —— 靠删键来切换模式。
- **魔法值没文档**:`prior_type` 论文附录没写,CLI 默认 `graph_scm` 是坏的;靠翻 TabICL 的 `train_stage{1,2,3}.sh` 才反推出必须是 `mix_scm`。
- **隐式副作用状态**:上面的 `out_embeddings`,加上满屏 `#todo fix for n_estimators>1`。
- **依赖不全**:`requirements.txt` 缺 numba、kditransform,得手动补。
- **运维暗雷**(不算代码库本身):单 GPU 不能并行(kernel-launch 排队,拖慢 ~3×);loky/joblib worker 不能 `pkill -9`(会污染 multiprocessing resource tracker,挂住其他进程)。

### 4.3 结论
**能复现、但不友好**的研究代码。核心实验脚本作者花过心思,但配置管线、模型 wrapper、依赖声明都是"论文发完就不管了"的质量。我们踩的坑(数据目录、`prior_type`、依赖)基本都源于**复现所需信息没写进代码或文档**,得靠读源码反推 —— 学术代码常态,但确实拖慢复现。

---

## 5. 交叉引用
- 复现总计划、服务器与运维:`exp4-exp6-repro-plan.md`
- Exp6 self-repair 出图:`plots/04a_skipping_layer.py::plot_skipping_finetuned`(需 c0/c2 两侧 `probing=="finetuned_decoder"`)
