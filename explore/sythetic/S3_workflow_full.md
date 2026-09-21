# 置换检验关系检测——完整工作流

# Permutation-Based Relationship Detection — Full Workflow

---

## 中文版

---

### 一、问题定义

**最终目标**：给定地球系统模型（ESM）的一个输入变量 x 和一个输出变量 y（共 n 个观测），判断 x 和 y 之间有没有统计依赖关系。

**"关系"的数学定义**：

设数据生成过程为：

```
Y = f(X) + σ(X)·ε
```

- f(X)：条件均值结构（y 的期望值如何随 x 变化）
- σ(X)：条件离散度（噪声大小如何随 x 变化）
- ε：标准化随机误差

**"没有关系"** 要求同时满足：
- f(X) = 常数（均值不随 x 变化）
- σ(X) = 常数（方差不随 x 变化）

即 P(Y|X) = P(Y)——知道 x 的值不提供关于 y 的任何信息。

只要 y 的分布以任何方式依赖 x——无论是均值变化、方差变化、还是两者同时变化——都算"有关系"。

---

### 二、假设检验框架

这是一个统计假设检验问题：

- **零假设 H₀**：X ⊥ Y（统计独立，没有任何关系）
- **备择假设 H₁**：X 和 Y 之间存在某种统计依赖

**为什么不直接看指标大小？** 因为即使 X ⊥ Y，有限样本下任何指标都不会精确等于 0。比如 500 个独立的 (x, y) 点，|Pearson r| 的期望值约为 1/√n ≈ 0.045，不是 0。需要统计框架来区分"真信号"和"随机波动"。

---

### 三、为什么需要合成数据

在真实 ESM 数据上，不知道真实答案——不知道哪些变量对真的有关系、哪些没有。无法验证方法是否可靠。

解决办法：构造**已知答案的合成数据**——
- 有些案例让 x 和 y 真的无关 → 方法应说"无关系"（否则是**误报**）
- 有些案例让 x 和 y 有已知关系 → 方法应说"有关系"（否则是**漏检**）

通过大量已知答案的案例验证方法可靠后，再应用到真实数据。

---

### 四、构造合成数据（S1）

#### 4.1 三旋钮生成模型

每个合成案例是一对 (x, y)，n = 500 个点，由三个可控旋钮生成：

```
y = f(x) + ε(x),    x ~ p(x)
```

**旋钮 1：f(x) — 均值信号**

f(x) 定义了 y 的期望值如何随 x 变化：
- 22 种函数族（线性、二次、正弦、阶梯、指数等各种形状）
- 1 种 Null：f(x) = 0，即没有均值信号

**旋钮 2：SNR — 信噪比**

控制信号相对噪声的强度，12 个水平：

```
0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, ∞
```

- SNR = 0.1 → 噪声是信号的 10 倍（极难检测）
- SNR = ∞ → 无噪声（极易检测）

**旋钮 3：ε(x) — 噪声展布模式**

控制噪声方差是否/如何随 x 变化，4 种模式：

| 模式 | 含义 |
|------|------|
| constant | 所有 x 处噪声方差相同（同方差） |
| increasing | x 越大噪声越大 |
| decreasing | x 越大噪声越小 |
| middle_high | 中间 x 处噪声最大 |

**旋钮 4：p(x) — x 的分布**

控制 x 的采样密度，8 种：even（均匀）、left_dense、right_dense、center_dense、clusters_2/3/4/5

#### 4.2 排列组合

23 函数 × 12 SNR × 4 噪声模式 × 8 x 分布 = **114,176 个案例**

#### 4.3 四类案例——已知答案

根据旋钮 1（f(x) 是否为 0）和旋钮 3（σ(x) 是否恒定），每个案例有一个已知的正确答案：

| 类别 | f(x) | σ(x) | 有关系吗？ | 正确判定 |
|------|------|------|---------|---------|
| **True Null** | = 0 | 恒定 | 无（y 和 x 完全独立） | not_detectable |
| **Mean-only** | ≠ 0 | 恒定 | 有（均值依赖 x） | detectable |
| **Variance-only** | = 0 | 随 x 变化 | 有（方差依赖 x） | detectable |
| **Mean+Variance** | ≠ 0 | 随 x 变化 | 有（均值+方差都依赖 x） | detectable |

关键点：Variance-only 不是"没有关系"。虽然 f(x) = 0（均值不变），但噪声大小依赖 x，y 的分布确实随 x 变化，这是真实的统计依赖。

---

### 五、描述性指标的初步探索（S2）

在设计检验之前，先对每对 (x, y) 计算大量描述性指标了解数据特征。S2 共计算了 **78 个指标**，分 12 组：

| 组 | 指标 | 数量 |
|---|---|---|
| 相关性 | Pearson r, Spearman ρ, 协方差 | 3 |
| 距离 | 距离协方差, 距离相关 (dcor) | 2 |
| MINE | MIC, MAS, MEV, MCN, MIC−r² | 5 |
| 斜率 | endpoint & polyfit × raw/std × 分段 + strength | 20 |
| 分箱 (等宽) | 振幅, η², 缓冲宽度, 有效分箱数 | 7 |
| 分箱 (等频) | 振幅, η², 缓冲宽度, 有效分箱数 | 7 |
| LOWESS | 残差SD, 振幅, R², 拐点, 总斜率 | 5 |
| GAM | 残差SD, 振幅, R², 拐点, 总斜率 | 5 |
| X 覆盖度 | KS距离, 分箱CV | 2 |
| 分布比较 | KS + Wasserstein × 等宽/等频 | 4 |
| y-SD 归一化 | y_sd + 16 个归一化比率 | 17 |
| 其他 | n_valid | 1 |

---

### 六、从 78 个指标中筛选 4 个用于置换检验

78 个指标不能全部放进置换检验——需要逐组评估。筛选依据有三层。

#### 6.1 第一层：无关系时的期望表现

置换检验的核心是比较"观测值"和"零假设下的值"。一个好的检测指标应该：无关系时接近 0，有关系时远离 0，且基线偏差小、方差稳定。

| 组 | 数量 | 代表指标 | 无关系时期望 | 筛选结果 |
|---|---|---|---|---|
| **相关性** | 3 | \|Pearson r\|, \|Spearman ρ\|, 协方差 | r, ρ ≈ 0 (~1/√n ≈ 0.045) | ✅ 选 \|r\|, \|ρ\|。协方差淘汰：受 y 尺度影响 |
| **距离** | 2 | 距离协方差, dcor | dcor ≈ 0（正偏 ~0.05–0.08） | ✅ 选 dcor。距离协方差淘汰：受尺度影响 |
| **MINE** | 5 | MIC, MAS, MEV, MCN, MIC−r² | MIC 正偏严重（Null 均值 ~0.16） | ❌ 全组淘汰。基线高→微弱信号淹没 |
| **斜率** | 20 | endpoint/polyfit × 各段 | ≈ 0，但**方差大** | ❌ 全组淘汰。方向可正可负，不稳定 |
| **分箱 (等宽)** | 7 | 振幅, η², 缓冲宽度, 分箱数 | η² ≈ 0 (~k/n ≈ 0.02)；缓冲宽度**不为 0** | ✅ 选 η²。缓冲宽度与关系无关 |
| **分箱 (等频)** | 7 | 同上 | 同上 | ❌ 与等宽冗余 |
| **LOWESS** | 5 | R², 残差SD, 振幅等 | R² ≈ 0 | ❌ 候选，看计算量 |
| **GAM** | 5 | R², 残差SD, 振幅等 | R² ≈ 0（略正偏） | ❌ 候选，看计算量 |
| **X 覆盖度** | 2 | KS距离, 分箱CV | 与 y 无关 | ❌ 度量 x 本身，不是 x-y 关系 |
| **分布比较** | 4 | KS + Wasserstein | 取决于分组方式 | ❌ 需分组决策，不如 dcor 直接 |
| **y-SD 归一化** | 17 | 各指标/y_sd | 是已有指标的缩放版 | ❌ 不是独立指标 |
| **其他** | 1 | n_valid | 与关系无关 | ❌ 度量数据质量 |

关于 X 覆盖度：虽然测试案例中有 x 分布不均匀的情况，但 X 覆盖度指标度量的是"x 本身分布均不均匀"，而不是"x 和 y 有没有关系"。置换检验打乱的是 y，x 保持不变——所以 x 的分布特征在观测值和 500 次置换中完全相同，零分布是一个常数，没有区分力。x 分布对检测力的影响通过置换检验自动适应（每个案例用自己的 x 生成零分布）。

#### 6.2 第二层：计算成本

置换检验需要对每个案例重复计算 500 次。114,176 个案例 × 501 次 ≈ 5,700 万次指标计算。

| 指标 | 单次复杂度 | 可向量化 | 114K × 501 总耗时（估算） |
|------|-----------|---------|----------------------|
| \|Pearson r\| | O(n) | ✅ 批量矩阵运算 | ~5 分钟 |
| \|Spearman ρ\| | O(n log n) | ✅ 批量 | ~8 分钟 |
| η² | O(n) | ✅ 批量 | ~7 分钟 |
| dcor | O(n²) | ❌ 逐案例循环 | **~15 小时** |
| MIC | O(n² · B) | ❌ 外部 C 库 | **~367 小时** |
| LOWESS R² | O(n²) 迭代 | ❌ | **~50+ 小时** |
| GAM R² | O(n) 迭代拟合 | ❌ | **~30+ 小时** |

MIC / LOWESS / GAM 计算量太大，且第一层筛选已发现 MIC 基线偏高。

#### 6.3 第三层：信息互补性

剩余候选（|r|, |ρ|, dcor, η²）之间是否冗余？

| 指标对 | 零分布相关性 | 覆盖差异 |
|-------|------------|---------|
| \|r\| ↔ \|ρ\| | ~0.95（高冗余） | ρ 对单调非线性更敏感；两者都极快，保留不增加成本 |
| \|r\| ↔ dcor | ~0.60 | dcor 覆盖非线性 + 方差依赖 |
| \|r\| ↔ η² | ~0.27–0.44 | η² 对非单调分段函数更敏感 |
| dcor ↔ η² | ~0.30–0.40 | dcor 擅长方差依赖，η² 擅长均值响应 |

η² 与其他三个相关性最低，提供最多独立信息。

#### 6.4 四个指标的互补关系

不同"高低组合"能推断出不同类型的关系：

| |r| | |ρ| | dcor | η² | 说明 | 关系类型举例 |
|---|---|---|---|---|---|
| 高 | 高 | 高 | 高 | 四个一致高 | 线性（y = ax + b） |
| 中 | 高 | 高 | 高 | \|r\| 弱于 \|ρ\| → 非线性但单调 | 指数、对数、幂函数 |
| 低 | 低 | 高 | 高 | 相关系数都抓不到 → 非单调 | 二次（U 形）、正弦 |
| 低 | 低 | 高 | 低 | 只有 dcor 高 → 均值不变但分布变了 | 方差随 x 变化（异方差） |
| 低 | 低 | 低 | 高 | 只有 η² 高 → 分箱后才看到差异 | 阶梯函数、局部突变 |
| 低 | 低 | 低 | 低 | 全低 → 没有可检测的依赖 | True Null（x ⊥ y） |

#### 6.5 最终选定

| 指标 | 来源组 | 为什么选 | 角色 |
|------|-------|---------|------|
| \|Pearson r\| | 相关性 | 线性检测器，计算极快，基线干净 | 线性关系最灵敏 |
| \|Spearman ρ\| | 相关性 | 单调非线性检测器，计算极快 | 补充非线性单调关系 |
| dcor | 距离 | 任意依赖，**唯一对方差依赖敏感** | 覆盖 Variance-only 类别 |
| η² (等宽分箱) | 分箱 | 分段均值响应，与其他三个零分布相关性最低 | 提供最多独立信息 |

四者覆盖：**线性 → 单调 → 任意非线性 → 方差依赖**。

#### 6.6 计算量

| 阶段 | 指标 | 方法 | 耗时 |
|------|------|------|------|
| Phase 1 | \|r\| + \|ρ\| + η² | 向量化批量 | ~20 分钟 |
| Phase 2 | dcor | 逐案例循环 (n×n 距离矩阵) | ~15 小时 |
| Phase 3 | Z-score + 联合检验 | 向量化 | ~1 分钟 |
| **总计** | | | **~15.5 小时** |

---

### 七、检测方法（S3）

S3 对全部 114,176 个案例做置换检验，分三个 notebook 并行处理不同指标组：

| Notebook | 指标 | 数量 | 覆盖 |
|----------|------|------|------|
| S3_permutation_test | \|r\|, \|ρ\|, dcor, η² | 4 核心 | 联合检验主体 |
| S3_permutation_test_all_metrics | 协方差, 斜率, 分箱, 分布比较, dcov | 30 + 1 + 4 = 35 | 扩展验证 |
| S3_permutation_mine | MIC, MAS, MEV, MCN | 4 | MINE 指标 |
| **合计** | | **43** | |

#### 7.1 计算观测值

对每个案例的原始 (x, y)（y 没有打乱）计算选定的指标。这些就是**观测值（observed values）**——真实数据里 x 和 y 的关联强度。

#### 7.2 置换检验——构建零分布

**问题**：如果 H₀ 为真（x ⊥ y），指标会是什么样？

**方法**：模拟 H₀ 为真时的指标值。

做法——对每个案例：
1. 保持 x 不变，**打乱 y 的顺序** → 得到 y_perm（打乱后 x 和 y 必然独立，H₀ 成立）
2. 对 (x, y_perm) 计算同样的指标
3. 重复 500 次 → 每个指标得到 500 个"零假设下的值"

这 500 个值就是**零分布（null distribution）**——"零"指的是"零假设"（null hypothesis），不是数值 0。零分布告诉我们"在没有关系的世界里，指标值通常在什么范围"。

**然后比较**：拿观测值和零分布比较。如果观测值远超零分布的范围，说明真实数据的关联不太可能是随机产生的 → 拒绝 H₀ → 有关系。

| | 来源 | 含义 |
|---|---|---|
| **观测值** | 原始 (x, y)，y 没有打乱 | 真实数据里 x 和 y 的关联强度 |
| **零分布** (500 个值) | (x, y_perm)，y 打乱了 | "没有关系"时指标能随机波动到多大 |

#### 7.3 Z-score 标准化

4 个指标的零分布尺度不同（如 dcor 零分布中位数 ~0.067，η² 的 ~0.015），不能直接比较。标准化为 Z-score：

```
Z_m = (M_obs − median(M_null)) / IQR(M_null)
```

- median(M_null) = 500 个零值的中位数
- IQR(M_null) = Q75 − Q25（四分位距）
- IQR 过小时退化为 std × 1.35

**为什么用 median/IQR 而不是 mean/std？** 因为这些指标都是非负绝对值，零分布右偏（偏度 ≈ 1），median/IQR 更稳健。

标准化后：Z ≈ 0 表示"和零假设下一样"，Z 越大 = 偏离零假设越远。

#### 7.4 联合检验

将 4 个 Z-score 合并为一个统计量：

```
T_joint = max(Z_|r|, Z_|ρ|, Z_dcor, Z_η²)
```

取 max 的逻辑：只要任何一个指标偏离零假设，就说明存在关系。不需要预先知道关系的形式。

#### 7.5 计算 p 值

把同样的 max 操作应用于零分布，构造联合零分布：

```
对每次置换 k = 1, ..., 500：
    T_null(k) = max(Z_|r|(k), Z_|ρ|(k), Z_dcor(k), Z_η²(k))

p = (T_null ≥ T_obs 的次数 + 1) / (500 + 1)
```

p 值的含义：**如果 x 和 y 真的无关，观测到当前这么大（或更大）T 值的概率**。

- p 很小 → 在无关系的假设下极不可能出现这种结果 → 拒绝 H₀ → 有关系
- p 很大 → 在无关系的假设下这种结果很正常 → 无法拒绝 H₀

这个 p 值是**精确的**（非参数的），并且天然控制了多重检验——因为零分布本身就是 4 个指标联合的，已经考虑了指标间的相关性。

最小可达 p 值 = 1/501 ≈ 0.002。

#### 7.6 分类判定

| p 值 | 分类 | 含义 |
|------|------|------|
| ≤ 0.05 | **detectable** | 拒绝 H₀，判定有关系 |
| 0.05 < p < 0.10 | **uncertain** | 不确定 |
| ≥ 0.10 | **not_detectable** | 无法拒绝 H₀，没检测到关系 |

---

### 八、验证方法可靠性（S4 + S5）

S4 的核心思路：单独生成**大量已知无关系的 Null 数据**，跑同样的置换检验，检查误报率（FPR）是否 ≈ 5%。

| Notebook | 指标 | Null 案例数 | 生成方式 |
|----------|------|-----------|---------|
| S4_Null_validate | 4 核心 | 4,000 | 简化版：normal 噪声 + constant spread |
| S4_null_validate_all_metrics | **全部 43 个** | 3,840 | 完整版：匹配 S1（4 noise × 4 spread × 8 x_dist） |

S4_null_validate_all_metrics 的生成与 S1 一致：
- 4 种噪声分布：normal, uniform, heavy_tail, skewed
- 4 种 spread_pattern：constant, increasing, decreasing, middle_high
- 8 种 x_distribution
- 128 种组合 × 30 个/组合 = 3,840 个 Null 案例

**误报验证**（4 核心指标，S4_Null_validate + S3 的 32 个 True Null = 4,032 个案例）：
- 方法误报 217 个
- **误报率 = 5.38%**，95% CI = [4.73%, 6.12%]
- α = 5% 在置信区间内 → 方法校准正确
- p 值在零假设下均匀分布 → p 值是可靠的

S4_null_validate_all_metrics 将进一步验证：全部 43 个指标在非正态噪声和异方差条件下的 FPR 是否仍然校准。

**检测力验证**（有关系的案例）：
- Mean-only：即使 SNR = 0.1（噪声是信号 10 倍），检测率 > 96%
- Variance-only：dcor 是主要检测器
- Mean+Variance：检测率略高于 Mean-only（方差依赖提供额外信号）

---

### 九、检测能力深入分析（S6）

- **Function × SNR 热力图**：哪些函数在哪些信噪比下最难检测
- **Metric ablation**：从已有零分布重构不同指标组合的联合检验，比较检测力
- **每个指标的不可替代贡献**：去掉该指标后多少案例从"检测到"变成"检测不到"
- **X 分布影响**：不同 x 分布对检测力的影响
- **阈值敏感性**：α 从 0.001 到 0.20，误报率和检测力如何变化

---

### 十、一句话总结

**定义 H₀: x ⊥ y → 构造 11 万个已知答案的合成 (x, y) → 从 78 个描述性指标中按"零假设下表现干净、计算可行、信息互补"筛选出 |r|、|ρ|、dcor、η² 四个核心指标（联合检验用），另外 39 个指标（扩展斜率/分箱/分布/MINE）单独做置换检验验证，共 43 个指标 → 对每个案例计算观测值，打乱 y 500 次构建零分布 → Z-score 标准化 → 取 max 联合检验 → p ≤ 0.05 判定"有关系" → 用 4000+ 个无关系案例（完整覆盖 4 噪声分布 × 4 spread 模式 × 8 x 分布）验证全部 43 个指标的误报率 → 方法可靠，可应用于真实 ESM 数据。**

---
---

## English Version

---

### 1. Problem Definition

**Goal**: Given an input variable x and an output variable y from an Earth System Model (ESM), with n observations, determine whether there is a statistically detectable relationship between x and y.

**Mathematical definition of "relationship"**:

Given the data-generating process:

```
Y = f(X) + σ(X)·ε
```

- f(X): conditional mean structure (how the expected value of y changes with x)
- σ(X): conditional dispersion (how noise magnitude changes with x)
- ε: standardized random error

**"No relationship"** requires both:
- f(X) = constant (mean does not depend on x)
- σ(X) = constant (variance does not depend on x)

i.e., P(Y|X) = P(Y) — knowing x provides no information about y.

Any dependence of y's distribution on x — whether through the mean, variance, or both — counts as a "relationship."

---

### 2. Hypothesis Testing Framework

- **Null hypothesis H₀**: X ⊥ Y (statistically independent, no relationship)
- **Alternative H₁**: X and Y are statistically dependent

**Why not just look at metric magnitudes?** Because with finite samples, metrics are never exactly 0 even when X ⊥ Y. For example, with n = 500 independent (x, y) points, E[|Pearson r|] ≈ 1/√n ≈ 0.045, not 0. A statistical framework is needed to distinguish "real signal" from "random fluctuation."

---

### 3. Why Synthetic Data

With real ESM data, the ground truth is unknown — we don't know which variable pairs truly have relationships. This makes it impossible to verify whether the method is reliable.

Solution: construct **synthetic data with known answers** —
- Some cases have x and y truly independent → method should say "no relationship" (otherwise: **false positive**)
- Some cases have a known x–y relationship → method should say "relationship" (otherwise: **false negative**)

Validate the method on these known-answer cases, then apply to real data.

---

### 4. Synthetic Data Construction (S1)

#### 4.1 Three-Knob Generation Model

Each synthetic case is an (x, y) pair, n = 500 points, generated by three controllable knobs:

```
y = f(x) + ε(x),    x ~ p(x)
```

**Knob 1: f(x) — mean signal**

Defines how y's expected value changes with x:
- 22 function families (linear, quadratic, sinusoidal, step, exponential, etc.)
- 1 Null: f(x) = 0, i.e., no mean signal

**Knob 2: SNR — signal-to-noise ratio**

Controls signal strength relative to noise, 12 levels:

```
0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, ∞
```

- SNR = 0.1 → noise is 10× the signal (very hard to detect)
- SNR = ∞ → no noise (trivial to detect)

**Knob 3: ε(x) — noise spread pattern**

Controls whether/how noise variance changes with x, 4 patterns:

| Pattern | Meaning |
|---------|---------|
| constant | Same noise variance everywhere (homoscedastic) |
| increasing | Noise grows with x |
| decreasing | Noise shrinks with x |
| middle_high | Noise is largest at mid-range x |

**Knob 4: p(x) — x distribution**

Controls sampling density of x, 8 types: even (uniform), left_dense, right_dense, center_dense, clusters_2/3/4/5

#### 4.2 Combinatorics

23 functions × 12 SNR × 4 noise patterns × 8 x-distributions = **114,176 cases**

#### 4.3 Four Categories — Known Answers

Based on Knob 1 (is f(x) zero?) and Knob 3 (is σ(x) constant?), each case has a known correct answer:

| Category | f(x) | σ(x) | Relationship? | Correct classification |
|----------|------|------|--------------|----------------------|
| **True Null** | = 0 | constant | No (y is independent of x) | not_detectable |
| **Mean-only** | ≠ 0 | constant | Yes (mean depends on x) | detectable |
| **Variance-only** | = 0 | varies with x | Yes (variance depends on x) | detectable |
| **Mean+Variance** | ≠ 0 | varies with x | Yes (both depend on x) | detectable |

Key point: Variance-only cases are not "no relationship." Although f(x) = 0 (mean is constant), noise magnitude depends on x, so y's distribution genuinely depends on x — this is real statistical dependence.

---

### 5. Descriptive Metrics Exploration (S2)

Before designing the test, S2 computes a large battery of descriptive metrics to understand the data. **78 metrics** across 12 groups:

| Group | Metrics | Count |
|---|---|---|
| Correlation | Pearson r, Spearman ρ, covariance | 3 |
| Distance | distance covariance, dcor | 2 |
| MINE | MIC, MAS, MEV, MCN, MIC−r² | 5 |
| Slopes | endpoint & polyfit × raw/std × segments + strength | 20 |
| Bins (equal-width) | amplitude, η², buffer widths, n_valid_bins | 7 |
| Bins (equal-count) | amplitude, η², buffer widths, n_valid_bins | 7 |
| LOWESS | residual SD, amplitude, R², sign changes, slope | 5 |
| GAM | residual SD, amplitude, R², sign changes, slope | 5 |
| X coverage | KS from uniform, bin count CV | 2 |
| Distribution comparison | KS + Wasserstein × equal-width/count | 4 |
| y-SD normalized | y_sd + 16 ratios | 17 |
| Misc | n_valid | 1 |

---

### 6. Selecting 4 Metrics from 78 for the Permutation Test

78 metrics cannot all be used in the permutation test. Selection proceeds through three filters.

#### 6.1 Filter 1: Behavior Under H₀ (No Relationship)

A good detection metric should be near 0 under H₀ with low bias and stable variance.

| Group | Count | Representative | Expected under H₀ | Selection |
|---|---|---|---|---|
| **Correlation** | 3 | \|Pearson r\|, \|Spearman ρ\|, covariance | r, ρ ≈ 0 (~1/√n ≈ 0.045) | ✅ Select \|r\|, \|ρ\|. Covariance dropped: scale-dependent |
| **Distance** | 2 | distance covariance, dcor | dcor ≈ 0 (positive bias ~0.05–0.08) | ✅ Select dcor. Distance covariance dropped: scale-dependent |
| **MINE** | 5 | MIC, MAS, MEV, MCN, MIC−r² | MIC has high positive bias (Null mean ~0.16) | ❌ All dropped. High baseline drowns weak signals |
| **Slopes** | 20 | endpoint/polyfit × segments | ≈ 0, but **high variance** | ❌ All dropped. Signed (±), unstable |
| **Bins (eq-width)** | 7 | amplitude, η², buffer width, bin count | η² ≈ 0 (~k/n ≈ 0.02); buffer width **≠ 0** | ✅ Select η². Buffer width reflects y's marginal, not x–y relationship |
| **Bins (eq-count)** | 7 | same | same | ❌ Redundant with equal-width |
| **LOWESS** | 5 | R², residual SD, etc. | R² ≈ 0 | ❌ Candidate, check cost |
| **GAM** | 5 | R², residual SD, etc. | R² ≈ 0 (slight positive bias) | ❌ Candidate, check cost |
| **X coverage** | 2 | KS distance, bin CV | Unrelated to y | ❌ Measures x-distribution, not x–y relationship |
| **Distribution** | 4 | KS + Wasserstein | Depends on grouping | ❌ Requires pre-grouping; dcor is more direct |
| **y-SD normalized** | 17 | metric / y_sd ratios | Rescaled versions | ❌ Not independent metrics |
| **Misc** | 1 | n_valid | Unrelated | ❌ Data quality, not relationship |

Regarding X coverage: although the test suite includes cases with non-uniform x distributions, X coverage metrics measure "how uniform is x itself," not "is there an x–y relationship." Since the permutation test shuffles y while keeping x fixed, x-distribution features are identical across the observed value and all 500 permutations — the null distribution would be a constant with no discriminating power. The effect of x-distribution on detection power is automatically handled by the permutation framework (each case's null distribution is generated using its own x).

#### 6.2 Filter 2: Computational Cost

The permutation test requires 114,176 cases × 501 evaluations ≈ 57 million metric computations.

| Metric | Complexity | Vectorizable | Estimated time (114K × 501) |
|--------|-----------|-------------|---------------------------|
| \|Pearson r\| | O(n) | ✅ batch matrix ops | ~5 min |
| \|Spearman ρ\| | O(n log n) | ✅ batch | ~8 min |
| η² | O(n) | ✅ batch | ~7 min |
| dcor | O(n²) | ❌ per-case loop | **~15 h** |
| MIC | O(n² · B) | ❌ external C library | **~367 h** |
| LOWESS R² | O(n²) iterative | ❌ | **~50+ h** |
| GAM R² | O(n) iterative fitting | ❌ | **~30+ h** |

MIC / LOWESS / GAM are too expensive, and Filter 1 already flagged MIC's high baseline.

#### 6.3 Filter 3: Complementarity (Low Redundancy)

Among the remaining candidates (|r|, |ρ|, dcor, η²):

| Metric pair | Null-distribution correlation | Coverage difference |
|------------|------------------------------|-------------------|
| \|r\| ↔ \|ρ\| | ~0.95 (high redundancy) | ρ more sensitive to monotonic nonlinearity; both are very fast, keeping both adds no cost |
| \|r\| ↔ dcor | ~0.60 | dcor covers nonlinear + variance dependence |
| \|r\| ↔ η² | ~0.27–0.44 | η² more sensitive to non-monotonic segmented functions |
| dcor ↔ η² | ~0.30–0.40 | dcor excels at variance dependence; η² excels at mean response |

η² has the lowest correlation with the other three, providing the most independent information.

#### 6.4 Complementary Roles of the Four Metrics

Different high/low patterns indicate different relationship types:

| |r| | |ρ| | dcor | η² | Pattern | Relationship type |
|---|---|---|---|---|---|
| High | High | High | High | All four high | Linear (y = ax + b) |
| Med | High | High | High | \|r\| < \|ρ\| → nonlinear but monotonic | Exponential, logarithmic, power |
| Low | Low | High | High | Correlations fail → non-monotonic | Quadratic (U-shape), sinusoidal |
| Low | Low | High | Low | Only dcor high → distribution shifts, not mean | Variance-only (heteroscedastic) |
| Low | Low | Low | High | Only η² high → binning reveals structure | Step function, local threshold |
| Low | Low | Low | Low | All low → no detectable dependence | True Null (x ⊥ y) |

#### 6.5 Final Selection

| Metric | Source group | Why selected | Role |
|--------|------------|-------------|------|
| \|Pearson r\| | Correlation | Linear detector, very fast, clean baseline | Most sensitive to linear relationships |
| \|Spearman ρ\| | Correlation | Monotonic nonlinear detector, very fast | Complements with nonlinear monotonic coverage |
| dcor | Distance | Arbitrary dependence, **only metric sensitive to variance-dependence** | Covers Variance-only category |
| η² (equal-width bins) | Bins | Segmented mean response, lowest null-correlation with the other three | Provides the most independent information |

Coverage spectrum: **linear → monotonic → arbitrary nonlinear → variance dependence**.

#### 6.6 Computation Budget

| Phase | Metrics | Method | Time |
|-------|---------|--------|------|
| Phase 1 | \|r\| + \|ρ\| + η² | Vectorized batch | ~20 min |
| Phase 2 | dcor | Per-case loop (n×n distance matrix) | ~15 h |
| Phase 3 | Z-score + joint test | Vectorized | ~1 min |
| **Total** | | | **~15.5 h** |

---

### 7. Detection Method (S3)

S3 runs the permutation test on all 114,176 cases, split across three notebooks for different metric groups:

| Notebook | Metrics | Count | Coverage |
|----------|---------|-------|----------|
| S3_permutation_test | \|r\|, \|ρ\|, dcor, η² | 4 core | Joint test backbone |
| S3_permutation_test_all_metrics | covariance, slopes, bins, distribution, dcov | 30 + 1 + 4 = 35 | Extended validation |
| S3_permutation_mine | MIC, MAS, MEV, MCN | 4 | MINE metrics |
| **Total** | | **43** | |

#### 7.1 Compute Observed Values

For each case's original (x, y) — with y unshuffled — compute the selected metrics. These are the **observed values**: the strength of association in the real data.

#### 7.2 Permutation Test — Build the Null Distribution

**Question**: If H₀ is true (x ⊥ y), what would the metrics look like?

**Method**: Simulate metric values under H₀.

For each case:
1. Keep x fixed, **shuffle y** → y_perm (after shuffling, x and y are necessarily independent — H₀ holds)
2. Compute the same metrics on (x, y_perm)
3. Repeat 500 times → 500 "H₀ values" per metric

These 500 values form the **null distribution** — "null" refers to the "null hypothesis," not the number zero. The null distribution tells us "what range of metric values is normal when there is no relationship."

**Then compare**: place the observed value against the null distribution. If the observed value far exceeds the null range, the association in the real data is unlikely to be random → reject H₀ → relationship exists.

| | Source | Meaning |
|---|---|---|
| **Observed value** | Original (x, y), y unshuffled | Association strength in the real data |
| **Null distribution** (500 values) | (x, y_perm), y shuffled | How large the metric can get by random chance alone |

#### 7.3 Z-Score Normalization

The four metrics have different null-distribution scales (e.g., dcor null median ~0.067, η² null median ~0.015), so they cannot be compared directly. Normalize to Z-scores:

```
Z_m = (M_obs − median(M_null)) / IQR(M_null)
```

- median(M_null) = median of the 500 null values
- IQR(M_null) = Q75 − Q25 (interquartile range)
- If IQR < 1e-12, fall back to std × 1.35

**Why median/IQR instead of mean/std?** These metrics are non-negative absolute values, so their null distributions are right-skewed (skewness ≈ 1). Median/IQR is more robust.

After normalization: Z ≈ 0 means "same as under H₀"; larger Z = stronger departure from H₀.

#### 7.4 Joint Test

Combine the 4 Z-scores into a single statistic:

```
T_joint = max(Z_|r|, Z_|ρ|, Z_dcor, Z_η²)
```

Rationale for max: if *any* metric departs strongly from the null, a relationship exists. No need to pre-specify which type of dependence to look for.

#### 7.5 Compute p-value

Apply the same max operation to the null distributions to construct the joint null:

```
For each permutation k = 1, ..., 500:
    T_null(k) = max(Z_|r|(k), Z_|ρ|(k), Z_dcor(k), Z_η²(k))

p = (#{T_null(k) ≥ T_obs} + 1) / (500 + 1)
```

p-value meaning: **the probability of observing a T value this large (or larger) if x and y are truly independent**.

- p is small → this result is extremely unlikely under H₀ → reject H₀ → relationship exists
- p is large → this result is normal under H₀ → cannot reject H₀

This p-value is **exact** (non-parametric) and inherently controls for multiple testing — the null distribution is the joint distribution of all 4 metrics, already accounting for their correlation structure.

Minimum achievable p-value = 1/501 ≈ 0.002.

#### 7.6 Classification

| p-value | Classification | Meaning |
|---------|---------------|---------|
| ≤ 0.05 | **detectable** | Reject H₀, relationship exists |
| 0.05 < p < 0.10 | **uncertain** | Inconclusive |
| ≥ 0.10 | **not_detectable** | Cannot reject H₀, no relationship detected |

---

### 8. Validation (S4 + S5)

S4's core idea: independently generate a **large number of known-null cases**, run the same permutation test, and check whether the false positive rate (FPR) ≈ 5%.

| Notebook | Metrics | Null cases | Generation |
|----------|---------|-----------|------------|
| S4_Null_validate | 4 core | 4,000 | Simplified: normal noise + constant spread |
| S4_null_validate_all_metrics | **All 43** | 3,840 | Full S1-matching: 4 noise × 4 spread × 8 x_dist |

S4_null_validate_all_metrics generation matches S1:
- 4 noise distributions: normal, uniform, heavy_tail, skewed
- 4 spread patterns: constant, increasing, decreasing, middle_high
- 8 x_distributions
- 128 combinations × 30 per combo = 3,840 null cases

**False positive validation** (4 core metrics, S4_Null_validate + S3's 32 True Null = 4,032 cases):
- 217 were falsely flagged
- **FPR = 5.38%**, 95% CI = [4.73%, 6.12%]
- α = 5% falls within the CI → method is correctly calibrated
- p-values are uniformly distributed under H₀ → p-values are reliable

S4_null_validate_all_metrics further validates: whether FPR remains calibrated for all 43 metrics under non-normal noise and heteroscedastic conditions.

**Detection power** (signal cases):
- Mean-only: even at SNR = 0.1 (noise is 10× signal), detection rate > 96%
- Variance-only: dcor is the primary detector
- Mean+Variance: detection rate slightly higher than Mean-only (variance dependence provides additional signal)

---

### 9. Detection Power Analysis (S6)

- **Function × SNR heatmap**: which functions are hardest to detect at which SNR
- **Metric ablation**: reconstruct joint tests from stored null distributions for different metric subsets; compare detection power
- **Unique contribution per metric**: how many cases are lost if that metric is removed
- **X distribution influence**: detection rate by x-distribution
- **Threshold sensitivity**: FPR and power across α from 0.001 to 0.20

---

### 10. One-Sentence Summary

**Define H₀: x ⊥ y → construct 114K synthetic (x, y) cases with known answers → select |r|, |ρ|, dcor, η² from 78 descriptive metrics based on clean H₀ behavior, computational feasibility, and complementarity as the 4 core metrics for the joint test; additionally validate 39 more metrics (extended slopes/bins/distribution/MINE) via separate permutation tests, totaling 43 metrics → compute observed values, shuffle y 500 times to build null distributions → Z-score normalize → max joint test → p ≤ 0.05 = "relationship detected" → validate all 43 metrics on 4,000+ null cases (fully covering 4 noise distributions × 4 spread patterns × 8 x-distributions) → method is reliable for real ESM data.**
