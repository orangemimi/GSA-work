# 散点图构造工作流（修订版 v4）


## 核心逻辑

```
函数族自带标签 → 生成散点图 → 标签随函数族直接继承
```

关键原则：

1. 标签不是从 metrics 推导的，而是函数族设计时直接赋予的（ground truth by construction）
2. 不需要"操作化判据"来判定标签；导数、斜率比等只用于解释标签来源
3. Cubic 作为独立的 2 turning points 函数族；更复杂的多 turning points 函数统一归入 F22 Complex
4. Strength/SNR 不进入标签体系，作为实验控制因子
5. 数据生成是上游，下游分析（metrics-based / raw scatterplot-based）另行设计
6. 所有单调非线性函数族均包含正向和负向版本，确保方向维度的完整覆盖


---

## 一、生成模型：三旋钮

```
y = f(x) + ε(x)
x ~ p(x)
```

| 旋钮 | 控制对象 | 决定的特征 |
|------|---------|-----------|
| 旋钮 1: f(x) | 函数族及其参数 | Shape labels（见两层标签体系） |
| 旋钮 2a: σ₀ | 噪声大小 | SNR（实验控制因子，不是标签维度） |
| 旋钮 2b: σ(x) | 噪声空间结构 | Changing Spread |
| 旋钮 3: p(x) | x 采样分布 | Density label, Cluster label |

不设第四个旋钮（amplitude a）。原因：y = a·f(x) + ε 中 a 的效果在相关系数类 metric 中被 scale-invariance 消除，在斜率类 metric 中被 y_sd 归一化消除，剩余效果等价于调 SNR。单独加 a 只增加实验维度，不产生新的区分信息。


---

## 二、两层标签体系

### 第一层：函数族 ID（F01–F22 + Null）

这是分类的 ground truth，直接由实验设计决定，不可能标错。

每个函数族拥有独立编号。正向和负向版本各自独立编号（如 F03/F04 为 Power convex 的正/负方向版本）。F21 Cubic 单独表示 cubic / two-turning 形状；F22 Complex 表示更复杂的非单调、多 turning points 形状。

### 第二层：可解释属性标签

挂在每个函数族上的人类可读描述，设计时固定，实验中不再改动。

属性维度：

| 属性 | 取值 | 适用范围 |
|------|------|---------|
| Direction | positive / negative / none / local / mixed / N/A | 所有族 |
| Monotonicity | monotonic / non-monotonic | 所有族 |
| Linearity | linear / nonlinear / N/A | 所有族 |
| Curvature | convex / concave / mixed / N/A | 非线性族；不适用处填 N/A |
| Special shape | saturation / S-curve / threshold / peak / valley / local peak / local valley / cubic / two-turning / complex / no relationship / none | 所有族 |
| TP number | 0 / 1 / 2 / Multiple / N/A | 所有族 |

不是每个族都填满所有维度。空维度 = N/A。

### 命名约定

正/负方向版本的族名中，"convex/concave" 指该函数在正方向时的曲率。负方向版本的实际曲率在标签表中标注（翻转后 convex ↔ concave）。

### 函数族标签总表

| Index | Function family | Equation | Direction | Linearity | Monotonicity | Curvature | Special shape | TP number |
|-------|----------------|----------|-----------|-----------|--------------|-----------|---------------|-----------|
| F01 | Linear positive | y = ax + b, a > 0 | positive | linear | monotonic | N/A | none | 0 |
| F02 | Linear negative | y = ax + b, a < 0 | negative | linear | monotonic | N/A | none | 0 |
| F03 | Power convex positive | y = xᵖ, p > 1 | positive | nonlinear | monotonic | convex | none | 0 |
| F04 | Power convex negative | y = −xᵖ, p > 1 | negative | nonlinear | monotonic | concave | none | 0 |
| F05 | Power concave positive | y = xᵖ, 0 < p < 1 | positive | nonlinear | monotonic | concave | none | 0 |
| F06 | Power concave negative | y = −xᵖ, 0 < p < 1 | negative | nonlinear | monotonic | convex | none | 0 |
| F07 | Saturation positive | y = 1 − exp(−kx) | positive | nonlinear | monotonic | concave | saturation | 0 |
| F08 | Saturation negative | y = −(1 − exp(−kx)) | negative | nonlinear | monotonic | convex | saturation | 0 |
| F09 | Log positive | y = log(1 + ax) | positive | nonlinear | monotonic | concave | none | 0 |
| F10 | Log negative | y = −log(1 + ax) | negative | nonlinear | monotonic | convex | none | 0 |
| F11 | Exponential positive | y = aˣ | positive | nonlinear | monotonic | convex | none | 0 |
| F12 | Exponential negative | y = −aˣ | negative | nonlinear | monotonic | concave | none | 0 |
| F13 | S-curve positive | logistic (increasing) | positive | nonlinear | monotonic | mixed | S-curve | 0 |
| F14 | S-curve negative | logistic (decreasing) | negative | nonlinear | monotonic | mixed | S-curve | 0 |
| F15 | Threshold positive | piecewise, b > a | positive | nonlinear | monotonic | N/A | threshold | 0 |
| F16 | Threshold negative | piecewise, b < a | negative | nonlinear | monotonic | N/A | threshold | 0 |
| F17 | Quadratic peak | y = −a(x−c)² + d | none / local | nonlinear | non-monotonic | N/A | peak (Inverted U-shape) | 1 |
| F18 | Quadratic valley | y = a(x−c)² + d | none / local | nonlinear | non-monotonic | N/A | valley (U-shape) | 1 |
| F19 | Spike | narrow peak | none / local | nonlinear | non-monotonic | N/A | local peak | 1 |
| F20 | L-shaped / narrow valley | piecewise valley | none / local | nonlinear | non-monotonic | N/A | local valley | 1 |
| F21 | Cubic | cubic polynomial | none / mixed | nonlinear | non-monotonic | mixed | cubic / two-turning | 2 |
| F22 | Complex | Double Gaussian, oscillation, etc. | none / mixed | nonlinear | non-monotonic | mixed | complex | Multiple |
| Null | No relationship | noise only | N/A | N/A | N/A | N/A | no relationship | N/A |

### 标签使用原则

族内参数变体标签一律相同。F07 Saturation positive 不管 k=0.5 还是 k=20，标签都是 {positive, nonlinear, monotonic, concave, saturation, 0 TP}。参数值 k 作为 metadata 记录，不进入标签。

k=0.5 的 saturation 在观测层面可能与 linear 不可区分——这不是标签错误，而是实验需要发现的退化边界。

正/负方向版本是独立的族，标签不同（至少 Direction 和 Curvature 不同），因此分别编号。

### 多粒度分析

两层标签支持灵活聚合，下游分析可在任意粒度上进行：

- 最细粒度：F01 vs F02 vs ... vs F22 vs Null
- 中间粒度：按属性组合分组（如所有 monotonic concave positive 合并）
- 最粗粒度：monotonic-linear / monotonic-nonlinear / 1-TP / 2-TP / complex
- 方向粒度：忽略正/负方向，合并同形状族（如 F03+F04 合并为 "Power convex"）


---

## 三、旋钮 1：函数族库

### 参数变体设计原则

每个族的参数变体保持密集覆盖，尽可能多地生成 case。参数范围应足够宽，自然覆盖三个区间：

| 区间 | 含义 | 分析价值 |
|------|------|---------|
| 原型区 | 特征明确，不可能与其他族混淆 | 建立 fingerprint 的 baseline |
| 中间区 | 特征可见但不极端 | 测试 metric 在典型条件下的表现 |
| 退化区 | 接近退化为更简单的形状 | 发现类间退化边界 |

三区不是采样约束（不是说每区只取一个值），而是分析框架——事后用来解读参数变体上的 metric 行为。

### 正/负方向版本的生成

所有单调非线性族（F03–F16）以正/负成对出现。负方向版本通过 y → −y 变换生成，参数变体与正方向版本完全相同。

### 已知退化关系

以下函数族在参数极端时会在观测层面趋近另一个族：

| 族 | 参数推向 | 退化为 |
|----|---------|--------|
| F03 Power convex positive | p → 1 | F01 Linear positive |
| F04 Power convex negative | p → 1 | F02 Linear negative |
| F05 Power concave positive | p → 1 | F01 Linear positive |
| F06 Power concave negative | p → 1 | F02 Linear negative |
| F07 Saturation positive | k → 0 | F01 Linear positive |
| F08 Saturation negative | k → 0 | F02 Linear negative |
| F09 Log positive | a → 0 | F01 Linear positive |
| F10 Log negative | a → 0 | F02 Linear negative |
| F11 Exponential positive | base → 1 | F01 Linear positive |
| F12 Exponential negative | base → 1 | F02 Linear negative |
| F13 S-curve positive | k → ∞ | F15 Threshold positive |
| F14 S-curve negative | k → ∞ | F16 Threshold negative |
| F15 Threshold positive | δ → ∞ | F13 S-curve positive（或 F01 Linear positive） |
| F16 Threshold negative | δ → ∞ | F14 S-curve negative（或 F02 Linear negative） |
| F07 Saturation positive | k → ∞ | F05 Power concave positive（强凹弱饱和区） |
| F08 Saturation negative | k → ∞ | F06 Power concave negative |
| F19 Spike | width → ∞ | F17 Quadratic peak |
| F20 L-shaped | width → ∞ | F18 Quadratic valley |


### 0 Turning Point（Monotonic）

**F01 Linear positive / F02 Linear negative**

```
y = ax + b

参数变体:
  slope |a| = 0.1, 0.3, 0.5, 1, 2, 5          (6 级)
  Direction: a > 0 (F01), a < 0 (F02)

每版 6 个配置, 共 12 个配置
F01 标签: positive, linear, monotonic, 0 TP
F02 标签: negative, linear, monotonic, 0 TP
```

**F03 Power convex positive / F04 Power convex negative**

```
F03: y = xᵖ    (x > 0)
F04: y = −xᵖ   (x > 0)

参数变体:
  p = 1.5, 2, 3, 5                             (4 级, f'' > 0 for positive)

每版 4 个配置, 共 8 个配置
F03 标签: positive, nonlinear, monotonic, convex, none, 0 TP
F04 标签: negative, nonlinear, monotonic, concave, none, 0 TP
退化: p → 1 时 F03 趋近 F01, F04 趋近 F02; p = 1.5 处于退化区
```

**F05 Power concave positive / F06 Power concave negative**

```
F05: y = xᵖ    (x > 0)
F06: y = −xᵖ   (x > 0)

参数变体:
  p = 0.2, 0.3, 0.5, 0.7                       (4 级, f'' < 0 for positive)

每版 4 个配置, 共 8 个配置
F05 标签: positive, nonlinear, monotonic, concave, none, 0 TP
F06 标签: negative, nonlinear, monotonic, convex, none, 0 TP
退化: p → 1 时 F05 趋近 F01, F06 趋近 F02; p = 0.7 处于退化区
```

**F07 Saturation positive / F08 Saturation negative**

```
F07: y = 1 − exp(−kx)
F08: y = −(1 − exp(−kx))

参数变体:
  k = 0.5, 1, 2, 3, 5, 8, 10, 15, 20          (9 级)

每版 9 个配置, 共 18 个配置
F07 标签: positive, nonlinear, monotonic, concave, saturation, 0 TP
F08 标签: negative, nonlinear, monotonic, convex, saturation, 0 TP
退化: k → 0 时趋近 Linear; k → ∞ 时趋近 Power concave
```

**F09 Log positive / F10 Log negative**

```
F09: y = log(1 + ax)
F10: y = −log(1 + ax)

参数变体:
  a = 1, 5, 10, 20, 50                         (5 级)

每版 5 个配置, 共 10 个配置
F09 标签: positive, nonlinear, monotonic, concave, none, 0 TP
F10 标签: negative, nonlinear, monotonic, convex, none, 0 TP
退化: a → 0 时趋近 Linear
与 Saturation 的关键区别: log 永远不会真正饱和到斜率=0
```

**F11 Exponential positive / F12 Exponential negative**

```
F11: y = aˣ
F12: y = −aˣ

参数变体:
  base a = 1.5, 2, 5, 10                       (4 级)
  x range: [0,1], [0,3], [0,5]                 (3 级)

每版 12 个配置, 共 24 个配置
F11 标签: positive, nonlinear, monotonic, convex, none, 0 TP
F12 标签: negative, nonlinear, monotonic, concave, none, 0 TP
退化: a → 1 时趋近 Linear
x range 决定曲线展开程度，range 小时接近线性
```

**F13 S-curve positive / F14 S-curve negative**

```
F13: y = L₁ + (L₂ − L₁) / (1 + exp(−k(x − c)))     (递增)
F14: y = −[L₁ + (L₂ − L₁) / (1 + exp(−k(x − c)))]  (递减)

两端渐近于 L₁ 和 L₂，但永远不真正到达（斜率趋近零但 ≠ 零）

参数变体:
  steepness k = 5, 8, 12, 20, 30               (5 级)
  transition center c = 0.3, 0.5, 0.7           (3 级)

每版 15 个配置, 共 30 个配置
F13 标签: positive, nonlinear, monotonic, mixed curvature, S-curve, 0 TP
F14 标签: negative, nonlinear, monotonic, mixed curvature, S-curve, 0 TP
退化: k → ∞ 时趋近 Threshold; k → 0 时趋近 Linear
关键特征: f'(x) 处处连续，两端渐近但不存在真正的 f'(x)=0 平坦段
```

**F15 Threshold positive / F16 Threshold negative**

```
F15 (b > a, 向上跳变):
  y = { a,                              x < c − δ
      { a + (b−a)/(2δ)·(x − c + δ),    c−δ ≤ x ≤ c+δ
      { b,                              x > c + δ

F16 (b < a, 向下跳变): 同上但 b < a

参数变体:
  state gap |b − a| = 0.3, 0.5, 1, 2            (4 级)
  transition width δ = 0.005, 0.01, 0.03, 0.05  (4 级)
  position c = 0.3, 0.5, 0.7                    (3 级)

每版 48 个配置, 共 96 个配置
F15 标签: positive, nonlinear, monotonic, threshold, 0 TP
F16 标签: negative, nonlinear, monotonic, threshold, 0 TP
退化: δ → ∞ 时趋近 S-curve 或 Linear
关键特征: 过渡带外 f'(x) = 0（真正的稳态段），与 S-curve 的核心区别
ESM 物理含义: tipping point，系统从一个 regime 翻转到另一个
```


### 1 Turning Point

**F17 Quadratic peak (Inverted U-shape) / F18 Quadratic valley (U-shape)**

```
∩ (peak):  y = −a(x − c)² + d
∪ (valley): y = a(x − c)² + d

参数变体:
  curvature a = 1, 4, 10, 20                    (4 级)
  position c = 0.2, 0.35, 0.5, 0.65, 0.8        (5 级)
  type: ∩ (F17) / ∪ (F18)

每版 20 个配置, 共 40 个配置
F17 标签: nonlinear, non-monotonic, peak (Inverted U-shape), 1 TP
F18 标签: nonlinear, non-monotonic, valley (U-shape), 1 TP
a 控制 turning 的锐度; c 偏离中心时大部分数据在一侧
```

**F19 Spike / F20 L-shaped (narrow peak/valley)**

```
Spike:    piecewise, narrow peak
L-shaped: piecewise, narrow valley

参数变体:
  peak/valley width: 0.02, 0.05, 0.1, 0.2       (4 级)
  position: 0.3, 0.5, 0.7                        (3 级)

每版 12 个配置, 共 24 个配置
F19 标签: nonlinear, non-monotonic, local peak, 1 TP
F20 标签: nonlinear, non-monotonic, local valley, 1 TP
退化: width → ∞ 时 F19 趋近 F17 Quadratic peak, F20 趋近 F18 Quadratic valley
width 极窄时大部分数据看不到 turning point
```


### 2 Turning Points

**F21 Cubic**

```
y = a(x − r1)(x − r2)(x − r3) + d

参数变体:
  root position: left, center, right                   (3 级)
  root spacing: narrow, medium, wide                   (3 级)
  type: M / W, controlled by sign of a                 (2 级)

共 18 个配置
标签: none / mixed, nonlinear, non-monotonic, mixed curvature, cubic / two-turning, 2 TP
```


### Multiple Turning Points

**F22 Complex non-monotonic functions**

所有 case 共享同一个标签：**none / mixed, nonlinear, non-monotonic, mixed curvature, complex, Multiple TP**。

设计原则：保留代表性函数作为压力测试。在真实 ESM 应用中，遇到全局复杂的 input-output 关系时，正确策略是分段识别——将 x 定义域切分为多个区间，每个区间内落入已有的 0 TP、1 TP 或 2 TP 框架。

代表性函数池：

```
Double Gaussian:       y = A₁·exp(−(x−μ₁)²/σ₁²) + A₂·exp(−(x−μ₂)²/σ₂²)
                       2 个配置（等高 / 不等高）

Pure oscillation:      y = A·sin(ωx)
                       2 个配置（低频 ω=2π / 高频 ω=8π）

Oscillation + trend:   y = bx + A·sin(ωx)
                       2 个配置（弱趋势 / 强趋势）

Damped oscillation:    y = A·exp(−λx)·sin(ωx)
                       1 个配置

Growing oscillation:   y = A·exp(λx)·sin(ωx)
                       1 个配置

Varying frequency:     y = A·sin(ωx(1 + x))
                       1 个配置

共 9 个配置
标签（统一）: none / mixed, nonlinear, non-monotonic, mixed curvature, complex, Multiple TP
```


### Null Baseline

与 x 完全独立的噪声，用于建立"无功能性关系"的 ground truth。

旋钮 2b（spread_pattern）对 Null 生效：噪声散布可随 x 变化（异方差），但条件均值始终为零。这会产生扇形或菱形散点图，视觉上看似有"结构"但不存在功能性关系。

旋钮 2a（SNR）对 Null 不适用：没有信号，无法定义信噪比。Null 在 case 生成时不与 SNR 交叉，固定记录为 ∞。

```
y = σ(x) · ε,    E[y|x] = 0

参数变体（噪声分布，所有分布标准化至 mean=0, variance=1）:
  normal:      ε ~ N(0, 1)                              基线，对称轻尾
  uniform:     ε ~ Uniform(−√3, √3)                     有界，无尾
  heavy_tail:  ε ~ t(df=3) / √3                         重尾，产生离群点
  skewed:      ε ~ Exponential(1) − 1                    右偏，非对称

σ(x) 由 spread_pattern 决定（σ₀ = 1.0）

共 4 个配置
标签: no relationship
```


### 旋钮 1 汇总

| 类别 | 族 | 配置数 |
|------|-----|--------|
| 0 TP Monotonic | F01–F16 | 206 |
| 1 TP | F17–F20 | 64 |
| 2 TP | F21 | 18 |
| Complex | F22 | 9 |
| Null | — | 4 |
| **合计** | | **301** |


---

## 四、旋钮 2：ε(x) 噪声结构

### 旋钮 2a：噪声大小（SNR）

```
SNR = Var(f(x)) / Var(ε)

SNR levels: 0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, ∞(无噪声)
共 12 级

实现方式:
  给定 f(x), 先在 x ~ Uniform(0,1) 上计算 Var(f(x))
  然后 σ_ε = sqrt(Var(f(x)) / SNR)
  ε ~ N(0, σ_ε), constant across x
```

### 旋钮 2b：噪声空间结构（heteroscedasticity）

```
Constant spread:     σ(x) = σ₀                                      (baseline)
Increasing spread:   σ(x) = σ₀ · (0.2 + 1.6x)                      (末端噪声大)
Decreasing spread:   σ(x) = σ₀ · (1.8 − 1.6x)                      (前端噪声大)
Middle-high spread:  σ(x) = σ₀ · (0.3 + 1.4·exp(−(x−0.5)²/0.02))

σ₀ 由 SNR 决定，乘以的系数控制噪声的空间分布
系数设计原则: E[σ(x)] ≈ σ₀，保证总噪声量级不变
```


---

## 五、旋钮 3：x 的采样分布

### 采样分布标签体系

| Distribution label | Sampling design | Representation | Density label | Cluster label |
|---|---|---|---|---|
| Even density | Even | x ~ Uniform(0, 1) | even | no clusters |
| Uneven density | Left-dense | x ~ Beta(2, 5) | uneven, left-dense | no clusters |
| Uneven density | Right-dense | x ~ Beta(5, 2) | uneven, right-dense | no clusters |
| Uneven density | Center-dense | x ~ Beta(5, 5) | uneven, center-dense | no clusters |
| Clustered density | Clusters | Uniform mixture (formula-based) | uneven | clusters |

### Clustered density 的生成公式

将多种 cluster 数量合并为统一的参数化模型，`n_clusters` 作为可选参数。

```
参数:
  n_clusters = 2, 3, 4, 5
  occupancy = 0.4（所有 cluster 占据 [0,1] 区间的总比例）

每个 cluster 的位置和宽度由公式生成:
  cluster_width = occupancy / n_clusters
  gap_width = (1 - occupancy) / (n_clusters + 1)
  cluster_i: Uniform(lo_i, hi_i)
    lo_i = gap_width × (i + 1) + cluster_width × i
    hi_i = lo_i + cluster_width

每个数据点以 90% 概率分配到某个 cluster（等概率），以 10% 概率从 Uniform(0,1) 背景采样。
background_frac = 0.1，保证 cluster 间有少量散点，更接近真实数据的聚集模式。
```

示例（occupancy = 0.4）:

| n_clusters | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 | Cluster 5 |
|---|---|---|---|---|---|
| 2 | [0.200, 0.400] | [0.600, 0.800] | — | — | — |
| 3 | [0.150, 0.283] | [0.433, 0.567] | [0.717, 0.850] | — | — |
| 4 | [0.120, 0.220] | [0.340, 0.440] | [0.560, 0.660] | [0.780, 0.880] | — |
| 5 | [0.100, 0.180] | [0.280, 0.360] | [0.460, 0.540] | [0.640, 0.720] | [0.820, 0.900] |


---

## 六、生成方案（全因子设计）

### 交叉设计：旋钮 1 × 旋钮 2a × 旋钮 2b × 旋钮 3

有信号族（F01–F22）做完全交叉，不做降维或代表性子集采样。Null 族不与 SNR 交叉（SNR 不适用），仅与旋钮 2b 和旋钮 3 交叉。

```
有信号族:
  297 函数配置 (F01–F22)
   × 12 SNR 水平
   ×  4 噪声结构 (constant / increasing / decreasing / middle-high)
   ×  8 采样分布 (even / left-dense / right-dense / center-dense / clusters n=2,3,4,5)
   = 114,048 个 case

Null 族:
    4 噪声分布配置 (normal / uniform / heavy_tail / skewed)
   ×  4 噪声结构 (constant / increasing / decreasing / middle-high)
   ×  8 采样分布
   = 128 个 case

合计: 114,048 + 128 = 114,176 个 case
每个 case 生成 R = 100 次随机实现
总计: 114,176 × 100 = 11,417,600 个散点图
```

### 因子明细

| 旋钮 | 因子 | 水平数 | 取值 |
|------|------|--------|------|
| 旋钮 1 | 函数配置 | 301 | F01–F22 (297) + Null (4) 全部参数变体 |
| 旋钮 2a | SNR | 12 | 0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, ∞（仅适用于有信号族） |
| 旋钮 2b | 噪声结构 | 4 | Constant / Increasing / Decreasing / Middle-high |
| 旋钮 3 | 采样分布 | 8 | Even / Left-dense / Right-dense / Center-dense / Clusters (n=2,3,4,5) |

### 生成总量

```
有信号族: 297 × 12 × 4 × 8 = 114,048 case
Null 族:    4 ×  1 × 4 × 8 =     128 case
合计:                         114,176 case
× 100 repeats              = 11,417,600 个散点图
```


---

## 七、固定参数

```
样本量:    n = 500 per case（固定，不作为实验因子）
x 归一化:  所有 case 的 x 映射到 [0, 1]
y 归一化:  生成后 y 不做归一化，保留 raw range
重复次数:  每个 case 生成 R = 100 次随机实现
随机种子:  seed = case_id × 1000 + r（保证可复现）
```


---

## 八、工作流：生成 → 标注

```
对每个 case:

  1. 设定三个旋钮的参数
     → 选函数族 + 参数值, 选 SNR, 选 x 分布

  2. 赋予标签
     → 标签直接从函数族 ID 和属性表继承（见第二节），不做任何计算
     → 参数值作为 metadata 记录（如 k=0.5, SNR=10, x~Uniform）

  3. 生成 R=100 次随机实现
     → 每次 n=500 个 (x, y) 数据点
     → 存储: 散点数据 + 标签 + metadata
```
