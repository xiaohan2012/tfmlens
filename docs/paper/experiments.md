# "Is One Layer Enough?" — 实验方法精确描述

**论文**：Balef et al., ICML 2026
https://arxiv.org/abs/2605.06510
**目标**：记录 6 个实验的精确计算过程，统一符号。

---

## 符号约定

| 符号 | 含义 |
|------|------|
| $L$ | 模型总层数（如 TabPFN v1 有 12 层） |
| $\ell \in \{0, 1, \ldots, L-1\}$ | 层索引 |
| $h_\ell(x_i) \in \mathbb{R}^d$ | 样本 $x_i$ 经过第 $\ell$ 层后的 embedding |
| $y_i \in \{0, 1\}$ | 样本 $x_i$ 的标签（二分类） |
| $\mathcal{S}$ | support set（训练集） |
| $\mathcal{Q}$ | query set（测试集） |
| $d(\cdot, \cdot)$ | 距离函数（默认 cosine distance） |
| $\hat{p}^\ell_i$ | 第 $\ell$ 层对样本 $i$ 的预测概率 |

---

## 六个实验与主旨的关系

**主旨**：TFM 的多层结构在做"**同一件事反复精炼**"，而非"**逐层专门化的不同任务**"。因此，单个 transformer block 的表达能力已经足够，只需迭代使用，不需要多种不同的专用层。

六个实验分两类，构成"理解 → 验证"的叙事弧线：

**A. 诊断型（理解层在做什么）**

| 实验 | 核心问题 | 与主旨的关系 |
|------|----------|------------|
| Exp 1 嵌入相似性 | 各层表示有多相似？ | 建立"层间存在冗余"的基础观察 |
| Exp 2 分离间隔 | 类别可分性如何随层变化？ | 说明层在做"逐步精炼"而非"专门化任务" |
| Exp 3 探针分类器 | 层 $i$ 的信息能预测层 $j$ 的输出吗？ | 进一步量化层间信息冗余 |
| Exp 4 Logit Lens | 每层单独解码能得到多高的性能？ | 定位"几层之后性能饱和" |

**B. 干预型（直接检验主旨）**

| 实验 | 核心问题 | 与主旨的关系 |
|------|----------|------------|
| Exp 5 层消融 | 跳过/重复/交换层，性能掉多少？ | 直接证明层可以互换，存在冗余 |
| Exp 6 Looped | 1 层循环 6 次 vs 6 个独立层？ | **核心证据**：同一计算迭代 ≈ 多样化层 |

Exp 6 是最直接的主旨证据，Exp 5 提供因果干预支撑，Exp 1-4 为 Exp 5/6 提供**动机**（观察到冗余现象，驱动后续干预实验）——注意因果干预（Exp 5/6）才是更强的解释，Exp 1-4 是相关性观察而非因果。

---

## Exp 1 — Embedding Similarity（嵌入相似度）

**目的**：论文的核心问题是"TFM 的深度是否必要"——如果每层都在学本质上不同的变换，那每层不可替代；如果相邻层的表示空间高度相似，说明模型在同一个空间里做迭代精化，深度在很大程度上是冗余的。Exp 1 是最直接的观察：不做任何干预，只看相邻层的 embedding 有多像，为后续的因果实验（Exp 5 跳层、Exp 6 自我修复）建立直觉预期。具体来看，是否形成 block structure——即连续多层的表示几乎不变——是"冗余深度"最直观的信号。

**计算步骤**：对每对层 $(\ell, \ell')$，计算两个指标：

### 1a. 平均绝对 Cosine Similarity

对所有样本行取平均（$N = |\mathcal{S}| + |\mathcal{Q}|$，每个 token = 一个样本行，即经过 ICL/row-attention 后的行级表示；对于有 feature-axis attention 的模型，feature 维度在进入 row-attention 之前已被聚合，所以此处的 $h_\ell(x_i) \in \mathbb{R}^d$ 始终是行级的）：

$$\text{CosSim}(h_\ell, h_{\ell'}) = \frac{1}{N} \sum_{i=1}^{N} \frac{h_\ell(x_i) \cdot h_{\ell'}(x_i)}{\|h_\ell(x_i)\|_2 \, \|h_{\ell'}(x_i)\|_2}$$

### 1b. Linear CKA

设 $X = [h_\ell(x_1), \ldots, h_\ell(x_N)]^\top \in \mathbb{R}^{N \times d}$，$Y = [h_{\ell'}(x_1), \ldots, h_{\ell'}(x_N)]^\top \in \mathbb{R}^{N \times d}$（均列中心化）：

$$\text{CKA}(X, Y) = \frac{\| X^\top Y \|_F^2}{\| X^\top X \|_F \cdot \| Y^\top Y \|_F + \varepsilon}$$

其中 $\varepsilon$ 为数值稳定项。CKA 对各向同性缩放不变，但对非各向同性缩放敏感（可能出现高 CosSim 低 CKA 的情况，说明某些 attention head 在做维度级别的缩放）。

**实现**（[`util.py:819`](https://github.com/amirbalef/is_one_layer_enough/blob/main/Experiments/util.py#L819-L820)）：

```python
# 外层循环：train_layer = ℓ，内层循环：eval_layer = ℓ'
# train_emb[i] = h_ℓ(x_i)，eval_emb[i] = h_ℓ'(x_i)，同一批 N 个样本，不同层

sim = cosine_similarity(train_emb, eval_emb)  # N×N 矩阵，[i,j] = cos(h_ℓ(x_i), h_ℓ'(x_j))
entry["cosine_similarity"] = float(np.mean(np.abs(np.diag(sim))))
# np.diag 取 i==j 项：同一样本跨两层的 cosine，对应公式中 h_ℓ(x_i)·h_ℓ'(x_i)
# np.mean 对所有样本求均值，对应公式中 (1/N)∑_i
entry["linear_cka"] = linear_cka(train_emb, eval_emb)  # util.py:504
```

**可视化**：$L \times L$ 热力图（上三角 = CKA，下三角 = CosSim），对所有数据集取平均。

**结论**：大模型（LimiX-16M、TabPFN(2.5)）形成明显的 block structure，小模型（TabPFN v1、TabICL）不明显。

<img src="figures/balef2026/exp1_similarity.png" alt="Figure 3 — Embedding Similarity" width="80%">

---

## Exp 2 — Separation Gap（分离间距）

**目的**：追踪模型每层对类别分离的贡献，量化分类信息随层的积累过程。

**计算步骤**：

定义类内对集合和类间对集合：

$$\mathcal{P}^{\text{within}} = \{(i,j) : y_i = y_j,\ i \neq j\}, \quad \mathcal{P}^{\text{between}} = \{(i,j) : y_i \neq y_j\}$$

从中各随机采样 100 对。对每层 $\ell$，先对 embedding 做 PCA（保留 95% 方差），然后在低维空间中计算距离：

$$D^\text{within}_\ell = \frac{1}{|\mathcal{P}^\text{within}|} \sum_{(i,j) \in \mathcal{P}^\text{within}} d\!\left(h_\ell(x_i),\ h_\ell(x_j)\right)$$

$$D^\text{between}_\ell = \frac{1}{|\mathcal{P}^\text{between}|} \sum_{(i,j) \in \mathcal{P}^\text{between}} d\!\left(h_\ell(x_i),\ h_\ell(x_j)\right)$$

$$\boxed{\Delta_\ell = D^\text{between}_\ell - D^\text{within}_\ell}$$

默认距离函数：cosine distance $d_\text{cos}(u,v) = 1 - \frac{u \cdot v}{\|u\|\|v\|}$。

**细节**：
- support 和 query 分别计算 $\Delta_\ell$
- 对于有 feature-axis attention 的模型（TabPFN v2、LimiX），还单独计算 feature embedding 和 label embedding 的 $\Delta_\ell$，展示"feature 先于 label 分离"的模式
- PCA 在跨所有层随机选 5000 个样本上拟合

**结论**：$\Delta_\ell$ 随层单调递增；label embedding 的 $\Delta$ 启动值已高（因为 support set 里有 label），feature embedding 稍滞后。

**图4 读图注记**：

- **蓝色曲线（label/feature token 分开）只在部分模型上出现**：TabPFN v2、v2.5、LimiX 有独立的 label token，可以拆开算；TabPFN v1 和 TabICL 所有信息混在同一 token 里，只有绿色的 "all" 曲线。

- **Support 曲线波动甚至下降的问题**：Query 的 $\Delta_\ell$ 基本单调递增，但 support 的曲线往往波动、甚至局部下降。论文的解释是"metric 是线性几何量，捕捉不到 embedding 的非线性变化"（基本等于没解释）。论文归因于"nonlinearity"，基本等于没解释。原因未明，是论文的一个未填坑。

<img src="figures/balef2026/exp2_separation_gap.png" alt="Figure 4 — Separation Gap" width="80%">

---

## Exp 3 — Probing Classifier（探针分类器）

**目的**：

Exp 2 只说了每层 embedding 在几何上越来越可分，但没有区分两种可能：
- **累积**：深层的表示是在浅层基础上叠加新信息，浅层信号被保留
- **替换**：深层重写了浅层的表示，浅层信号消失

Exp 3 通过跨层迁移矩阵来区分这两种情况——如果信息是累积的，浅层训的 probe 在深层上应该仍然好用（上三角偏高）；如果是替换，则只有对角线高。

**计算步骤**：

1. **数据划分**：把原始**训练集**对半分：
   - **Part 1**（前半）→ TFM 的 support set，标签放入 context
   - **Part 2**（后半）→ 作为 TFM 的 query 处理（标签对模型不可见）→ 用来**训练 probe**（$\mathcal{Q}^{\text{train}}$）
   - **原始测试集** → 同样作为 TFM 的 query → 用来**验证 probe**（$\mathcal{Q}^{\text{val}}$）

   之所以用 query embedding 而非 support embedding 来训 probe，是因为 support token 的输入里天然含有标签，embedding 会直接编码标签信息，probe 会虚高。

2. **训练**：对每一层 $i$，在 $\{h_i(x) : x \in \mathcal{Q}^{\text{train}}\}$ 上训练 logistic regression probe $f_i$。

3. **跨层迁移评估**：对每对层 $(i, j)$，用 $f_i$ 在 $\{h_j(x) : x \in \mathcal{Q}^{\text{val}}\}$ 上评估，得到 ROC-AUC 矩阵：

$$M[i, j] = \text{AUC}\!\left(f_i,\ \{h_j(x)\}_{x \in \mathcal{Q}^{\text{val}}}\right)$$

4. **归一化**：按列最大值归一化。

**$M[i,j]$ 的直觉**：用第 $i$ 层训的 probe，去测第 $j$ 层的 embedding，AUC 有多高。
- **高**：两层 embedding 空间"兼容"，$i$ 层学到的线性分类边界在 $j$ 层仍然成立
- **低**：两层不兼容——要么 $j$ 层信息比 $i$ 层更丰富（probe 不够用），要么两层用不同方式组织信息

**关键信号读法**：
- **上三角高**（$j > i$）：浅层 probe 在深层仍好用 → 信息**累积**，深层保留了浅层的结构
- **下三角低**（$j < i$）：深层 probe 在浅层上不好用 → 深层有浅层没有的东西，不可逆
- **只有对角线高**：每层各自编码不同信息，互不兼容 → **替换**而非累积

**结论**：存在长程不对称（程度因模型而异），支持"residual stream 是累积的"假说。

<img src="figures/balef2026/exp3_probing.png" alt="Figure 5 — Probing Classifiers" width="80%">

---

## Exp 4 — Tabular Logit Lens（表格 Logit 透镜）

**目的**：判断每层的 embedding 在功能上是否已经足以做出可靠预测（而不仅仅是线性可分）。

**为什么不直接用原始 decoder**：TFM 的 decoder 和最后一层的表示空间紧耦合，直接把它应用到中间层会产生大量 entropy（预测极不自信），信号失真。

**计算步骤**：

1. **为每层单独训练 decoder**：在 TabICL 合成 prior 生成的数据集上（同 TabICL 的预训练数据分布），对每层 $\ell$ 单独 fine-tune 一个 decoder $g_\ell$（从原始 decoder 初始化，继续训练）。

2. **推理时逐层解码**：对每个测试样本 $x$，在每层 $\ell$ 处：
$$\hat{p}^\ell(x) = g_\ell\!\left(h_\ell(x)\right)$$

3. **评估**：对每层计算 ROC-AUC：
$$\text{AUC}^\ell = \text{AUC}\!\left(\hat{p}^\ell,\ y\right)$$

**与 Exp 3 的区别**：

| | Exp 3 Probing | Exp 4 Logit Lens |
|---|---|---|
| 输出形状 | $L \times L$ 矩阵 | $L$ 维向量 |
| 研究的关系 | 层 $i$ 与层 $j$ 之间的信息兼容性 | 每层独立的预测能力 |
| 需要额外权重？ | 否 | 是（本地缺失，需 GPU 预训练） |
| 核心问题 | 信息是累积还是被替换？ | 有用信息在哪层已经形成？ |

**结论**：前几层 AUC 就已经急剧升高（有用信息早已形成），后续层的 AUC 基本持平。用原始 decoder 做 logit lens（不 fine-tune）的 AUC 始终落后，说明后期层在做"representation 对齐 decoder basis"而不是"生成新预测信息"。

**开放问题（待问更强的模型）**：

- Individual decoder（橙线）在 TabPFN v2 第5层左右就收敛了，后续层对橙线没有贡献。既然如此，为什么不直接训练一个更浅的模型，或者砍掉已训练模型的尾部层再 fine-tune decoder？

- 后期层（橙线收敛之后）到底在做什么？论文说是"prediction calibration"（对齐 decoder），但机制未解释清楚。

- 一个可能的担忧：橙线在推理时收敛不代表训练时后面的层没用——前5层能产生好的表示，可能正是因为端到端训练中后面的层提供了梯度压力。直接砍层再训是否等价？论文没有直接测试这个。

<img src="figures/balef2026/exp4_logit_lens.png" alt="Figure 6 — Tabular Logit Lens" width="80%">

---

## Exp 5 — Layer Ablation（层消融）

**目的**：通过因果干预直接测量每层对最终预测的贡献。

三种干预操作，每种分别对 $m \in \{0, \ldots, L-1\}$ 执行：

### 5a. Skip（跳层）

把第 $m$ 层从 forward pass 中删除，直接让第 $m-1$ 层的输出传给第 $m+1$ 层：

$$h^{\text{skip-}m}_\ell = \begin{cases} h_\ell(x) & \ell < m \\ h_{m-1}(x) & \ell = m \text{（跳过）} \\ f_\ell(h_{m-1}(x)) & \ell = m+1 \end{cases}$$

度量：最终 AUC 相对于 baseline（不干预）的变化 $\Delta\text{AUC} = \text{AUC}^{\text{skip-}m} - \text{AUC}^{\text{baseline}}$。

### 5b. Repeat（重复层）

在位置 $m$ 处执行两次第 $m$ 层（总层数变为 $L+1$）：

$$h^{\text{repeat-}m}_\ell = \begin{cases} h_\ell(x) & \ell < m \\ f_m(h_{m-1}(x)) & \ell = m \\ f_m(h^{\text{repeat-}m}_m(x)) & \ell = m+1 \text{（再执行一次）} \end{cases}$$

### 5c. Swap（交换层）

交换第 $m$ 层和第 $m+1$ 层的权重，其他层不变。

**结论**：
- Skip：$m = 0$ 时 AUC 暴跌（−37% on Bank_Customer_Churn），$m \geq 2$ 时几乎无影响
- Repeat：对 LimiX-16M 和 TabPFN v1 有轻微提升，支持"迭代精化"假说
- Swap：所有模型均受损，TFM 对层序比 LLM 更敏感

<img src="figures/balef2026/exp5_layer_ablation.png" alt="Figure 7 — Layer Ablation" width="80%">

---

## Exp 6 — Self-Repair（自我修复）

**目的**：区分"中后期层可以被跳过"是因为（a）这些层天然冗余（什么都没做），还是（b）后续层主动补偿了被跳层的计算。

**计算步骤**：结合 Exp 4 和 Exp 5。

1. 对每个跳层位置 $m$，执行 skip 干预。
2. 不只看最终层的 AUC，而是用 tabular logit lens 看跳层之后**每一层**的 AUC：

$$\text{AUC}^{\text{skip-}m}_\ell \quad \text{for all } \ell > m$$

3. 对比 baseline 轨迹 $\text{AUC}^\ell$，画出恢复曲线：
   - 如果 $\text{AUC}^{\text{skip-}m}_\ell \approx \text{AUC}^\ell$ 对于 $\ell \gg m$：**有自我修复**
   - 如果 $\text{AUC}^{\text{skip-}m}_\ell < \text{AUC}^\ell$ 持续到最后一层：**无自我修复**（该层有唯一功能）

**可视化**：
- 黑色实线：baseline 无干预的 layer-wise AUC
- 彩色实线（蓝→橙 = 早→晚跳）：干预后的 layer-wise AUC
- ×：被跳过的层
- 虚线：从被跳层之后第一层连到 baseline，视觉上标出"掉落 + 恢复"

**结论**：
- 跳第 0 层：无法恢复，后续层的 AUC 全程低于 baseline
- 跳中后期层：后续层的 AUC 恢复到 baseline，尤其 TabPFN v2 表现最明显

---

## 最终实验：Looped Transformer

**动机**：Exp 5（repeat 有提升）+ Exp 6（中后期层冗余但有自我修复能力）→ 如果层间做的是相似的迭代精化，是否可以只训练一层，循环执行 $K$ 次？

**对比设置**（均使用 nanoTabPFN 架构，6 层）：

| 模型 | 参数量 | 层结构 |
|------|--------|--------|
| nanoTabPFN（1 层） | 最小 | 单层，不循环 |
| nanoTabPFN（6 层） | 基准 | 标准 6 层 |
| **Looped nanoTabPFN** | 同 1 层 | 单层权重，forward 时循环执行 6 次 |

**训练**：Looped 版在训练时也循环 6 次（weights shared），从 scratch 训练。

**结论**：
- 1 层（不循环）< baseline 6 层（性能明显差）
- **Looped 6 次 ≈ 普通 6 层**（在 PMLBmini 和 TabArena 上接近）
- 解释：深度的主要作用是允许迭代计算，而不是每层学习本质上不同的变换

<img src="figures/balef2026/exp6_looped.png" alt="Figure 8 — Looped Transformer" width="80%">

**复现说明**：NanoTabPFN 训练代码**不在本 repo**，需要从外部 repo `https://github.com/automl/TFM-Playground/` 训练。本 repo 的 `configs/nanotabpfn/` 目录实际不存在。绘图脚本 `plots/05_nanotabpfn.py` 和 `plots/05_looped_tabicl.py` 存在，但 looped TabICL 对应的 config（c6/c7）未公开，该对比实验无法本地复现。

**批评与遗漏**：

1. **TabICL 的反例被忽略**：从 Figure 6 可以看到，TabICL 的 individual decoder（橙线）在进入 Row-wise interaction 第 1 层时就已经接近满分。Column embedder 阶段（$C_1, C_3, R_2$）已经完成了大部分信息提取，row-wise interaction 几乎只是收尾。这意味着 TabICL 天然地"1 层就够"，是比 NanoTabPFN 更直接的实验对象，论文完全没有讨论。

2. **Exp 6 的结论被标题过度泛化**：Figure 6 中 TabPFN(v2) 的橙线在第 1 层只有 ~0.6，到第 5-7 层才收敛——说明 TabPFN v2 架构单层明显不够。Exp 6 实际检验的是"looped 1 block × 6次 ≈ 6个独立 block"（循环架构的有效性），而不是"1 层表达能力天生足够"。这两件事被论文标题混在一起了。

| 命题                                 | 是否被 Exp 6 直接检验 |
| ---------------------------------- | -------------- |
| looped 1 block × 6 次 ≈ 6 个独立 block | ✓              |
| 1 层单次就能达到好效果                       | ✗（图里明显不够）      |
| TabPFN v2 架构天生适合 1 层               | ✗              |

---

## Extensions / ToDo

### 本地可跑（✅）vs 不可跑（❌）

| 实验 | 本地可跑？ | 状态 | 备注 |
|------|-----------|------|------|
| Exp 1 embedding similarity（c0） | ✅ | 已完成 | tabpfn_v1 + tabicl，10 个数据集 |
| Exp 2 separation gap（c0） | ✅ | 已完成 | tabpfn_v1，10 个数据集；tabpfn_v2/v2_5 也已跑完 |
| Exp 3 probing（c1） | ✅ | 已完成（smoke） | tabpfn_v1，10 个数据集，logistic regression only |
| Exp 4 tabular logit lens | ❌ | 无法复现 | finetuned decoder 权重未公开 |
| Exp 5 skip（c2） | ✅ | 已完成 | tabpfn_v1 × 363619，12 层全部跑完 |
| Exp 5 repeat（c3） | ✅ | 部分完成 | tabpfn_v1 × 363619，p0-p4；p5-p11 待补全 |
| Exp 5 swap（c4） | ✅ | 已完成 | tabpfn_v1 × 363619，已修复 config（缺 finetuned_decoder pop） |
| Exp 6 looped NanoTabPFN | ❌ | 无法复现 | 训练代码在外部 repo TFM-Playground；looped TabICL config 未公开 |
| LimiX 系列（2M/16M） | ❌ | 缺权重 | LimiX-2M.ckpt / LimiX-16M.ckpt 未下载 |

### 待在 GPU 服务器上完成
- **扩展 Exp 1 + Exp 2**：补全 limix_2m / limix_16m（需先下载权重）× 10 个 TabArena 数据集
  - 跑命令：`bash run_c0.sh <model>`（顺序跑，不并行；已有结果自动跳过）
  - 数据集：363619 363621 363671 363629 363626 363696 363682 363684 363700 363674
- **出图**：每个模型的 separation gap 均值曲线（对齐论文 Figure 4）
- **Exp 3 完整版**：config_c1（全部 probe 模型，较慢）
- **Exp 5 c3 补全**：tabpfn_v1 × 363619，p5-p11

---

## 论文核心假设

**关于泛化性**
- G1（模型）：现有 5 个模型（TabPFN v1/v2/v2.5、TabICL、LimiX）的结论能代表 TFM 范式——更新的模型（TabICL v2、TabFM、Mitra）未验证 → **对应研究方向三**
- G2/G3（任务+数据集）：结论基于 TabArena 49 个分类数据集——能否推广到回归任务和其他 benchmark 尚未验证 → 暂不做

**关于架构**
- A6：论文的分析框架主要为 row-level embedding 设计——对 TabICL column embedder 阶段的分析不系统 → 目前只有 TabICL 用 column embedder，样本量太少，等更多模型有类似架构时再讨论
- A7：NanoTabPFN 足以代表 TFM 普遍行为——论文自己承认结论不能直接外推到更大模型 → 接受，暂不深入

**关于实验设计**（风险最高）
- A8：单层干预结果能反映层的功能——AUC 下降可能来自分布偏移而非该层本身的重要性 → 接受，实验设计的固有局限
- A9：finetuned decoder 在 TabICL 合成数据上训练，能泛化到真实数据集 → **Action point**：等 finetuned decoder 权重公开后做验证实验
- A11：looped NanoTabPFN 的结论能推广到已训练的大模型——直接截断已训练模型是独立问题 → **对应研究方向一的核心实验**（本地可跑）

---

## 研究方向一：后期层的机制与必要性

**核心问题**：individual decoder 在第 5 层后收敛，后期层到底在做什么、能不能省？对应假设 A8、A11。

- **机制层**：attention attribution / gradient attribution 分析后期层的功能（calibration？decoder alignment？）
- **工程层**：
  - 连续多层 skip（如同时跳掉 layer 3-5），容忍度如何
  - 截断已训练模型的后几层直接用，性能损失多少（直接验证 A11）
  - 直接训练浅层模型 vs. 截断深层模型，差距反映后期层对前层的梯度压力（验证 A8）
  - Looped 模型推理时循环 $K' < K$ 次，accuracy-compute tradeoff 如何

---

## 研究方向二：TabICL 的特殊性

**核心问题**：TabICL 的 column embedder 已完成大部分信息提取，row-wise interaction 几乎只需 1 层——论文对此分析不系统（对应假设 A6）。

- "0 层 row-interaction"消融：直接用 column embedder 输出预测，性能下降多少（量化 column embedder 的实际贡献）
- Looped TabICL：1 个 row-wise block 循环多次 vs. 多个独立 block
- 专门针对 column embedder 阶段的 probing / separation gap 分析

---

## 研究方向三：更新模型的分析

**核心问题**：本文覆盖的模型均为 2024-2025 年的工作，更新的模型是否有相同的规律？对应假设 G1。

- TabICL v2
- Google TabFM
- Amazon Mitra
- 以及后续出现的其他 TFM

---

## Misc

- **层顺序敏感性**：Exp 5 swap 只测相邻层，跨距离 swap（如 layer 1 和 layer 10 互换）尚未测试
- **Support 曲线波动**（Exp 2 未解释）：控变实验——对 support set 做 label shuffle，看波动是否与标签信息有关
