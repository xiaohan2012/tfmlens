# tfm_lens —— 全新代码库设计与协作手册(会话交接文档)

> **用途**:这份文档是为了在**另一个 Claude Code session** 里无缝继续而写的。它保留了我们关于「从零重建一个干净代码库来复现论文 Exp4/5/6」的全部设计讨论、已定决策、待定岔路、以及协作方式。新会话读完这份文档即可直接接着干。
>
> **本库位置**:`/Users/hanxiao/code/tfmlens/`(本文件 = `DESIGN.md`,在仓库根)。
> **论文**:Balef et al. *Is One Layer Enough?* (ICML 2026)。
> **参考(旧)代码库**:`/Users/hanxiao/docs/ml-reading/code/is_one_layer_enough/`(能复现但不友好的研究代码,作为参考实现)。
> **论文材料(随库附带,新会话先读)**:`docs/paper/balef2026onelayer.md`(论文主页,含全部实验图 `docs/paper/assets/`)、`docs/paper/experiments.md`(六个实验的详细笔记)、`docs/paper/concepts/`(tabular-logit-lens / self-repair / layer-ablation / looped-transformer 四个机制概念页)。
> **相关文档**:`docs/old-code-exp4-walkthrough.md`(旧代码 Exp4 走读,随库附带);`/Users/hanxiao/docs/ml-reading/notes/exp4-exp6-repro-plan.md`(复现总计划与运维,留在 ml-reading)。

---

## 0. 目标与已定决策

**要造什么**:一个干净、易扩展的新库 `tfm_lens`,实现论文的三个实验:
- **Exp4**(大头):训练 per-layer decoder = "tabular logit lens" 工具本身。唯一花算力的,产出被 Exp5/6 复用的资产。
- **Exp5**:循环/重复层(looping)+ 用 logit lens 量效果。
- **Exp6**:self-repair = 跳过某层后,用 logit lens 量下游如何补偿。

**已拍板的决策**:
1. ✅ **只做分类**(classification only)。回归(regression)暂不做,接口留口子即可。
2. ✅ **全新独立库**,不在旧仓内重构。
3. ✅ **位置(已定)** `/Users/hanxiao/code/tfmlens/`。是否 `git init` 待定。
4. ✅ **协作方式:一步一步、模块化、测试驱动、接口先行。用户要全程参与,不要一次性写完所有代码。**

---

## 1. 核心洞察:三个实验共享同一个原语

拆开看,Exp4/5/6 只有一件事不同,其余全共享:

```
冻结骨干前向 → 抓每层 residual stream → [可选:干预某层] → 解码成预测
```

| 实验 | 干预 | 用什么解码 | 特有产物 |
|---|---|---|---|
| **Exp4** | 无 | **训练** per-layer decoder | `decoder_layer_*.pth`(核心资产) |
| **Exp5** | 循环/重复层 | 复用 Exp4 decoder | 循环 N 次曲线 |
| **Exp6** | 跳过某层 | 复用 Exp4 decoder | self-repair 曲线 |

**推论**:Exp4 是大头(唯一花算力 + 产出复用资产);新库应把「训好的 decoder」当**一等缓存产物**。Exp5/6 基本是「选一个 intervention + 调 logit_lens」十几行。

> ⚠️ 注意编号陷阱:旧仓 `plots/` 里文件编号(04a_skipping / 05_looped / 06_compute_cost)**≠ 论文实验编号**。self-repair(用户口中的 Exp6)对应 `plots/04a_skipping_layer.py::plot_skipping_finetuned`。

---

## 2. 为什么旧代码难扩展(重构动机的证据)

旧代码靠**改 vendored 模型源码**来抓中间表示 + 做干预,是**侵入式手术**。以 LimiX 为例,改动散落两个文件:

**A. `FoundationModels/Limix/limix/model/layer.py`(transformer block 本体)**
```python
# __init__ (:420-421) 埋缓存槽
self.out_embeddings = None
self.in_embeddings  = None
# forward 开头 (:650) 抓输入
self.in_embeddings = x.clone().detach().cpu()
# forward 残差路径里 (:672-679) 插侧信道 + 干预
self.component_contribution(sublayer_name, x, residual, eval_pos, task_info)   # 侧信道日志
x[:, :eval_pos] = x[:, :eval_pos] * task_info.get("w_"+sublayer_name+"_on_support", 1.0)  # 干预!
x[:, eval_pos:] = x[:, eval_pos:] * task_info.get("w_"+sublayer_name+"_on_query", 1.0)
# forward 结尾 (:685) 抓输出
self.out_embeddings = x.clone().detach().cpu()
```

**B. `sklearn/classifier.py` + `regressor.py`** —— 4~5 个 wrapper 方法读 `layer.out_embeddings`。

**两个关键事实**:
- **干预焊死在源码**:`config_base.py` 里的 `w_attn_sequence_on_query: 1.0` 那堆权重,靠 `task_info` 字典**穿透整个 forward** 实现;跳过一层 = 设成 `0.0`。为此 forward 签名也被改了(多了 `eval_pos`/`task_info`/`layer_idx`)。
- **每个模型各一份**:LimiX / TabPFN_v1 / TabICL 各有自己的 `layer.py`,加新模型 = 在它源码里**重做整台手术**。

→ 难扩展的根源不是接口差,是**逻辑长在了模型内脏里**。

---

## 3. 设计支点:用 hook 取代源码手术

旧代码的层是以 `transformer_encoder.layers[l]` **模块**形式可访问的(`classifier.py:180`),所以可以用 **PyTorch 原生 forward hook** 完全取代源码 patch。**同一套 hook 机制同时解决「抓」和「干预」,且模型无关。**

**库选型(已比较,已决策)**:用 **PyTorch 原生 hook + 一个 context manager,不引第三方库**。
- **transformer_lens**:❌ 否决。硬编码 GPT/Llama 类架构,塞不进 LimiX/TabPFN 的自定义架构(feature-attn + sample-attn + label token)。
- **nnsight**:⚠️ 能用但过度设计。为「抓 N 层 + skip/repeat」引重依赖 + trace 心智模型不划算。
- **原生**:✅ 需求窄且明确,~50 行,零依赖,完全可控。

分工:
- **抓表示** = `layer.register_forward_hook(...)` 记录输出,取代 `self.out_embeddings = ...`。
- **干预** = context manager 临时替换 `layer.forward`,退出即还原:
  - **skip** = forward 返回输入(恒等)
  - **repeat / loop N** = 包一层连调 N 次
  - **swap** = 互换两层 forward

---

## 4. 目录结构(用户已认可)

```
tfm_lens/
├── adapters/              # 唯一需要为新模型写代码的地方
│   ├── base.py            # ModelAdapter 抽象接口
│   ├── limix.py           # ~30 行
│   ├── tabpfn_v2.py
│   └── registry.py        # name → adapter 工厂(@register 装饰器)
├── core/
│   ├── capture.py         # forward-hook 抓 per-layer residual stream
│   ├── interventions.py   # skip / repeat / loop / swap(context manager)
│   └── logit_lens.py      # adapter + decoders → 每层预测
├── data/
│   └── prior.py           # on-the-fly mix_scm(就一处)
├── training/
│   └── train_decoders.py  # Exp4:旧 fine_tuning_exp.py 清洗版
├── experiments/
│   ├── exp4_train.py      # 薄驱动
│   ├── exp5_looping.py    # 薄驱动:loop 干预 + logit_lens
│   └── exp6_self_repair.py# 薄驱动:skip 干预 + logit_lens
├── eval/ + plotting/      # 共享打分与出图
├── configs/               # 每模型一个小文件,只放差异
└── tests/                 # pytest
```

---

## 5. 协作方式:每个模块三道闸

```
① 接口/契约  →  用户 review 拍板
② 测试(先写,红)  →  用户 review 覆盖是否对
③ 实现(让测试变绿)  →  用户 review 代码
```
每道闸停下等用户,不越界往下写。**不要一次性写完所有代码。**

---

## 6. 构建顺序(按依赖自底向上)

关键:**capture + interventions 可完全用"玩具模型"做 TDD,不需要 GPU / LimiX checkpoint。** 从这里起步,测试飞快。

| # | 模块 | 依赖 | 无 GPU 可测 |
|---|---|---|---|
| **1** | `adapters/base.py` 接口 + 玩具 fixture | 无 | ✅ 纯接口 |
| **2** | `core/capture.py` | 1 | ✅ 玩具模型 |
| **3** | `core/interventions.py` | 1 | ✅ 玩具模型 |
| **4** | `core/logit_lens.py` | 1,2 | ✅ 玩具 decoder |
| **5** | `adapters/limix.py`(第一个真模型) | 1 | ❌ 要 ckpt |
| **6** | `data/prior.py` | 无 | ⚠️ 慢,可 mock |
| **7** | `training/train_decoders.py`(Exp4) | 全部 | ❌ 要 GPU |
| **8** | `experiments/exp4/5/6` | 全部 | 集成 |

前 4 个模块 = 骨架,占设计 80%;后面基本是填 adapter + 接线。

**下一步(新会话从这里继续)**:走第 1 个模块 —— review 下面的 `ModelAdapter` 接口,解决 §8 的三个设计点,然后给 `core/capture.py` 写第一个(会失败的)测试。

---

## 7. 已提议的 `ModelAdapter` 接口(待 review)

```python
# tfm_lens/adapters/base.py
from abc import ABC, abstractmethod
import torch.nn as nn

class ModelAdapter(ABC):
    """把任意冻结的 tabular FM 适配成『可逐层读取 + 可干预』的对象。
    实现者只需指到模型的层列表 + 声明几个能力,不改模型源码。"""

    # ---- 能力声明(取代旧代码散落的 config bool)----
    label_token_index: int | None = -1   # 读哪个 token 的表示;None=非 label-token 架构
    needs_transpose:   bool = False       # decoder 是否要 (seq,batch,hidden)

    # ---- 必须实现 ----
    @property
    @abstractmethod
    def layers(self) -> list[nn.Module]:
        """可挂 hook / 可替换 forward 的层列表(通常是 nn.ModuleList)。"""

    @abstractmethod
    def decoder_template(self) -> nn.Module:
        """骨干自带的 decoder 头,会被 deepcopy 成 per-layer decoder。"""

    @abstractmethod
    def forward_frozen(self, X, y_train, eval_pos: int) -> None:
        """在 no_grad 下跑一次前向,仅为触发 hook;返回值不重要。"""

    # ---- 可选覆盖 ----
    def post_norm(self, emb):
        """decoder 前的归一化(如 LimiX 的 encoder_out_norm);默认恒等。"""
        return emb

    @property
    def n_layers(self) -> int:
        return len(self.layers)
```

参考:LimiX adapter 草图(实现时的样子)见本文件末尾附录,或旧走读文档。核心方法映射:
- `layers` → `self.model.transformer_encoder.layers`
- `decoder_template` → `self.model.cls_y_decoder`(旧 `get_decoder`)
- `post_norm` → `self.model.encoder_out_norm`(旧 `get_decoder_pre_norm`)
- `forward_frozen` → 旧 `model_inference`
- `label_token_index=-1` 取代旧 `separate_y_embeddings`;`needs_transpose` 同名

---

## 8. 待 review 的三个接口设计点(新会话先解决这个)

1. **`forward_frozen` 返回 `None`**:表示全靠 hook 抓,前向不返回表示。倾向**不返回**,让 `capture_layers` context manager 独占「抓」职责。隐患:「必须先 forward 再读 cache」的时序耦合还在(只是从源码挪到了 context manager 里)。→ 用户定。
2. **能力声明:类属性 vs 方法**:`label_token_index`/`needs_transpose` 用类属性;`post_norm` 用方法(要跑张量)。→ 这个划分 OK 吗?
3. **`layers` 返回 list vs ModuleList**:TabICL 是两段式 encoder+predictor,`layers` 可能要拼接两段。倾向返回**普通 list**,干预时靠「替换元素的 forward」而非「换 list」。→ 用户定。

---

## 9. core 模块草图(供第 2/3 步参考,尚未实现)

```python
# core/capture.py —— 抓每层 residual stream(取代所有 out_embeddings 手术)
from contextlib import contextmanager

@contextmanager
def capture_layers(adapter):
    cache, handles = [], []
    for layer in adapter.layers:
        handles.append(layer.register_forward_hook(
            lambda m, inp, out: cache.append((out[0] if isinstance(out, tuple) else out).detach())
        ))
    try:
        yield cache
    finally:
        for h in handles:
            h.remove()

# core/interventions.py —— skip,对所有模型通用(取代 task_info 权重穿透)
@contextmanager
def skip_layer(adapter, idx):
    layer = adapter.layers[idx]
    orig = layer.forward
    layer.forward = lambda x, *a, **k: x        # 恒等 = 跳过
    try:
        yield
    finally:
        layer.forward = orig
```

Exp6 最终会薄成十几行:
```python
# experiments/exp6_self_repair.py
adapter  = build_adapter("limix_2m", ckpt)
decoders = load_finetuned_decoders(adapter, weights_dir)   # Exp4 产物
for L in range(adapter.n_layers):
    with skip_layer(adapter, L), capture_layers(adapter) as cache:
        adapter.forward_frozen(X, y_train, eval_pos)
    preds = logit_lens(cache, decoders, adapter)
    record_self_repair(L, preds)
```

---

## 10. 关键参考(旧仓,复现细节的真相来源)

- Exp4 训练循环:`is_one_layer_enough/Experiments/fine_tuning_exp.py`(`main:343`, `run_on_micro_batches:217`, `train_step:130`)
- LimiX wrapper:`FoundationModels/Limix/limix/sklearn/classifier.py`(`:177` embeddings, `:243` inference, `:233` load_finetuned_decoders)
- 数据参数(已核对论文):`mix_scm`(70% MLP-SCM + 30% tree-SCM),`min/max_features=2/30`,`max_classes=10`,`max_seq_len=1024`,`train_size` 0.1~0.9。**`prior_type` 必须是 `mix_scm`**(CLI 默认 `graph_scm` 是坏的)。
- 训练规模:200 步 × 512 表 = 102,400 表;LimiX-2M = 12 层 → 13 个 decoder;`AdamW(lr=3e-5, wd=1e-4)`。
- 运维暗雷:单 GPU 不能并行(kernel-launch 排队 ~3× 慢);loky worker 不能 `pkill -9`(污染 resource tracker)。详见 `exp4-exp6-repro-plan.md`。
