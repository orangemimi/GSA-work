# 散点图构造工作流（修订版 v3）


## 核心逻辑

```
函数族自带标签 → 生成散点图 → 标签随函数族直接继承
```

关键原则：

1. 标签不是从 metrics 推导的，而是函数族设计时直接赋予的（ground truth by construction）
2. 不需要"操作化判据"来判定标签；导数、斜率比等只用于解释标签来源
3. Cubic 作为独立的 2 turning points 函数族；更复杂的多 turning points 函数统一归入 F15 Complex
4. Strength/SNR 不进入标签体系，作为实验控制因子
5. 数据生成是上游，下游分析（metrics-based / raw scatterplot-based）另行设计


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
| 旋钮 3: p(x) | x 采样分布 | Density, Clusters |

不设第四个旋钮（amplitude a）。原因：y = a·f(x) + ε 中 a 的效果在相关系数类 metric 中被 scale-invariance 消除，在斜率类 metric 中被 y_sd 归一化消除，剩余效果等价于调 SNR。单独加 a 只增加实验维度，不产生新的区分信息。


---

## 二、两层标签体系

### 第一层：函数族 ID（F01–F15 + Null）

这是分类的 ground truth，直接由实验设计决定，不可能标错。

F01–F15 各自拥有独立编号。F14 Cubic 单独表示 cubic / two-turning 形状；F15 Complex 表示更复杂的非单调、多 turning points 形状，具体函数形式作为生成时的实现细节。

### 第二层：可解释属性标签

挂在每个函数族上的人类可读描述，设计时固定，实验中不再改动。

属性维度：

| 属性 | 取值 | 适用范围 |
|------|------|---------|
| Direction | positive / negative / none / none / local / none / mixed | 所有族 |
| Monotonicity | monotonic / non-monotonic | 所有族 |
| Linearity | linear / nonlinear / none | 所有族 |
| Curvature | convex / concave / mixed / no relationship / N/A | 非线性族；不适用处填 N/A |
| Special shape | saturation / weak/no saturation / S-curve / threshold / peak / valley / local peak / local valley / cubic / two-turning / complex / none | 所有族 |
| TP number | 0 / 1 / 2 / Multiple / none | 所有族 |

不是每个族都填满所有维度。空维度 = N/A。

### 函数族标签总表

| Index | Function family | Equation/Presentation | Basic label | Direction | Linearity | Monotonicity | Curvature | Special shape | TP number |
|-------|-----------------|-----------------------|-------------|-----------|-----------|--------------|-----------|---------------|-----------|
| F01 | Linear positive | y = ax + b, a > 0 | slope a | positive | linear | monotonic | N/A | none | 0 |
| F02 | Linear negative | y = ax + b, a < 0 | slope a | negative | linear | monotonic | N/A | none | 0 |
| F03 | Power convex | y = x^p, p > 1 | exponent p | positive | nonlinear | monotonic | convex | none | 0 |
| F04 | Power concave | y = x^p, 0 < p < 1 | exponent p | positive | nonlinear | monotonic | concave | none | 0 |
| F05 | Saturation | y = 1 - exp(-kx) | rate k | positive | nonlinear | monotonic | concave | saturation | 0 |
| F06 | Log | y = log(1 + ax) | scale a | positive | nonlinear | monotonic | concave | weak/no saturation | 0 |
| F07 | Exponential | y = a^x | base a, x range | positive | nonlinear | monotonic | convex | none | 0 |
| F08 | S-curve | logistic form | steepness k, center c | positive | nonlinear | monotonic | mixed | S-curve | 0 |
| F09 | Threshold | piecewise state change | gap, width, position | positive | nonlinear | monotonic | N/A | threshold | 0 |
| F10 | Quadratic peak (Inverted U-shape) | y = -a(x-c)^2 + d | curvature a, position c | none / local | nonlinear | non-monotonic | N/A | peak (Inverted U-shape) | 1 |
| F11 | Quadratic valley (U-shape) | y = a(x-c)^2 + d | curvature a, position c | none / local | nonlinear | non-monotonic | N/A | valley (U-shape) | 1 |
| F12 | Spike | narrow peak | width, position | none / local | nonlinear | non-monotonic | N/A | local peak | 1 |
| F13 | L-shaped / narrow valley | piecewise valley | width, position | none / local | nonlinear | non-monotonic | N/A | local valley | 1 |
| F14 | Cubic | cubic polynomial | roots, sign, spacing, type = M/W | none / mixed | nonlinear | non-monotonic | mixed | cubic / two-turning | 2 |
| F15 | Complex non-monotonic functions | cubic, double Gaussian, oscillation, etc. | nonlinear, non-monotonic, multiply turning points | none / mixed | nonlinear | non-monotonic | mixed | complex | Multiple |
| Null | No relationship | noise only | none | N/A | none | N/A | no relationship | none | none |

### 标签使用原则

族内参数变体标签一律相同。F05 Saturation 不管 k=0.5 还是 k=20，标签都是 {positive, nonlinear, monotonic, concave, saturation, 0 TP}。参数值 k 作为 metadata 记录，不进入标签。

k=0.5 的 saturation 在观测层面可能与 linear 不可区分——这不是标签错误，而是实验需要发现的退化边界。

### 多粒度分析

两层标签支持灵活聚合，下游分析可在任意粒度上进行：

- 最细粒度：F01 vs F02 vs ... vs F15 vs Null
- 中间粒度：按属性组合分组（如所有 monotonic concave 合并）
- 最粗粒度：monotonic-linear / monotonic-nonlinear / 1-TP / 2-TP / complex


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

### 已知退化关系

以下函数族在参数极端时会在观测层面趋近另一个族：

| 族 | 参数推向 | 退化为 |
|----|---------|--------|
| F03 Power convex | p → 1 | F01 Linear |
| F04 Power concave | p → 1 | F01 Linear |
| F05 Saturation | k → 0 | F01 Linear |
| F06 Log | a → 0 | F01 Linear |
| F07 Exponential | base → 1 | F01 Linear |
| F08 S-curve | k → ∞ | F09 Threshold |
| F09 Threshold | δ → ∞ | F08 S-curve（或 Linear） |
| F05 Saturation | k → ∞ | F04 Power concave（强凹弱饱和区） |
| F12 Spike | width → ∞ | F10 Quadratic peak (Inverted U-shape) |
| F13 L-shaped | width → ∞ | F11 Quadratic valley (U-shape) |


### 0 Turning Point（Monotonic）

**F01 Linear positive / F02 Linear negative**

```
y = ax + b

参数变体:
  slope |a| = 0.1, 0.3, 0.5, 1, 2, 5          (6 级)
  Direction: a > 0 (Positive), a < 0 (Negative)  (2 级)

共 12 个配置
标签: [positive/negative], linear, monotonic, 0 TP
```

**F03 Power convex**

```
y = xᵖ    (x > 0)

参数变体:
  p = 1.5, 2, 3, 5                             (4 级, f'' > 0)

共 4 个配置
标签: positive, nonlinear, monotonic, convex, 0 TP
退化: p → 1 时趋近 Linear; p = 1.5 处于退化区
```

**F04 Power concave**

```
y = xᵖ    (x > 0)

参数变体:
  p = 0.2, 0.3, 0.5, 0.7                       (4 级, f'' < 0)

共 4 个配置
标签: positive, nonlinear, monotonic, concave, 0 TP
退化: p → 1 时趋近 Linear; p = 0.7 处于退化区
```

**F05 Saturation**

```
y = 1 - exp(-kx)

参数变体:
  k = 0.5, 1, 2, 3, 5, 8, 10, 15, 20          (9 级)

共 9 个配置
标签: positive, nonlinear, monotonic, concave, saturation, 0 TP
退化: k → 0 时趋近 Linear
```

**F06 Log**

```
y = log(1 + ax)

参数变体:
  a = 1, 5, 10, 20, 50                         (5 级)

共 5 个配置
标签: positive, nonlinear, monotonic, concave, no saturation, 0 TP
退化: a → 0 时趋近 Linear
与 F05 的关键区别: log 永远不会真正饱和到斜率=0
```

**F07 Exponential**

```
y = aˣ

参数变体:
  base a = 1.5, 2, 5, 10                       (4 级)
  x range: [0,1], [0,3], [0,5]                 (3 级)

共 12 个配置
标签: positive, nonlinear, monotonic, convex, 0 TP
退化: a → 1 时趋近 Linear
x range 决定曲线展开程度，range 小时接近线性
```

**F08 S-curve**

```
y = L₁ + (L₂ - L₁) / (1 + exp(-k(x - c)))

两端渐近于 L₁ 和 L₂，但永远不真正到达（斜率趋近零但 ≠ 零）

参数变体:
  steepness k = 5, 8, 12, 20, 30               (5 级)
  transition center c = 0.3, 0.5, 0.7           (3 级)

共 15 个配置
标签: positive, nonlinear, monotonic, mixed curvature, S-curve, 0 TP
退化: k → ∞ 时趋近 F09 Threshold; k → 0 时趋近 Linear
关键特征: f'(x) 处处连续，两端渐近但不存在真正的 f'(x)=0 平坦段
```

**F09 Threshold**

```
y = { a,                              x < c - δ       (稳态 A, 斜率=0)
    { a + (b-a)/(2δ)·(x - c + δ),    c-δ ≤ x ≤ c+δ   (过渡带)
    { b,                              x > c + δ       (稳态 B, 斜率=0)

参数变体:
  state gap |b - a| = 0.3, 0.5, 1, 2            (4 级)
  transition width δ = 0.005, 0.01, 0.03, 0.05  (4 级)
  position c = 0.3, 0.5, 0.7                    (3 级)

共 48 个配置
标签: positive, nonlinear, monotonic, threshold, 0 TP
退化: δ → ∞ 时趋近 S-curve 或 Linear
关键特征: 过渡带外 f'(x) = 0（真正的稳态段），与 S-curve 的核心区别
ESM 物理含义: tipping point，系统从一个 regime 翻转到另一个
```


### 1 Turning Point

**F10 Quadratic peak (Inverted U-shape) / F11 Quadratic valley (U-shape)**

```
∩ (peak):  y = -a(x - c)² + d
∪ (valley): y = a(x - c)² + d

参数变体:
  curvature a = 1, 4, 10, 20                    (4 级)
  position c = 0.2, 0.35, 0.5, 0.65, 0.8        (5 级)
  type: ∩ / ∪                                    (2 级)

共 40 个配置
标签: nonlinear, non-monotonic, [peak (Inverted U-shape) / valley (U-shape)], 1 TP
a 控制 turning 的锐度; c 偏离中心时大部分数据在一侧
```

**F12 Spike / F13 L-shaped (narrow peak/valley)**

```
Spike:    piecewise, narrow peak
L-shaped: piecewise, narrow valley

参数变体:
  peak/valley width: 0.02, 0.05, 0.1, 0.2       (4 级)
  position: 0.3, 0.5, 0.7                        (3 级)

共 24 个配置
标签: nonlinear, non-monotonic, [local peak / local valley], 1 TP
退化: width → ∞ 时趋近 F10 Quadratic peak (Inverted U-shape) / F11 Quadratic valley (U-shape)
width 极窄时大部分数据看不到 turning point
```


### 2 Turning Points

**F14 Cubic**

```
y = a(x - r1)(x - r2)(x - r3) + d

参数变体:
  root position: left, center, right                   (3 级)
  root spacing: narrow, medium, wide                   (3 级)
  type: M / W, controlled by sign of a                 (2 级)

共 18 个配置
标签: none / mixed, nonlinear, non-monotonic, mixed curvature, cubic / two-turning, 2 TP
```


### Multiple Turning Points

**F15 Complex non-monotonic functions**

所有 case 共享同一个标签：**none / mixed, nonlinear, non-monotonic, mixed curvature, complex, Multiple TP**。

设计原则：保留代表性函数作为压力测试。在真实 ESM 应用中，遇到全局复杂的 input-output 关系时，正确策略是分段识别——将 x 定义域切分为多个区间，每个区间内落入已有的 0 TP、1 TP 或 2 TP 框架。

代表性函数池：

```
Double Gaussian:       y = A₁·exp(-(x-μ₁)²/σ₁²) + A₂·exp(-(x-μ₂)²/σ₂²)
                       2 个配置（等高 / 不等高）

Pure oscillation:      y = A·sin(ωx)
                       2 个配置（低频 ω=2π / 高频 ω=8π）

Oscillation + trend:   y = bx + A·sin(ωx)
                       2 个配置（弱趋势 / 强趋势）

Damped oscillation:    y = A·exp(-λx)·sin(ωx)
                       1 个配置

Growing oscillation:   y = A·exp(λx)·sin(ωx)
                       1 个配置

Varying frequency:     y = A·sin(ωx(1 + x))
                       1 个配置

共 9 个配置
标签（统一）: none / mixed, nonlinear, non-monotonic, mixed curvature, complex, Multiple TP
```


### Null Baseline

```
y ~ N(μ, σ)    与 x 完全无关

共 1 个配置
标签: no relationship
```


### 旋钮 1 汇总

| 类别 | 族 | 配置数 |
|------|-----|--------|
| 0 TP Monotonic | F01–F09 | 109 |
| 1 TP | F10–F13 | 64 |
| 2 TP | F14 | 18 |
| Complex | F15 | 9 |
| Null | — | 1 |
| **合计** | | **201** |


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
Decreasing spread:   σ(x) = σ₀ · (1.8 - 1.6x)                      (前端噪声大)
Middle-high spread:  σ(x) = σ₀ · (0.3 + 1.4·exp(-(x-0.5)²/0.02))

σ₀ 由 SNR 决定，乘以的系数控制噪声的空间分布
系数设计原则: E[σ(x)] ≈ σ₀，保证总噪声量级不变
```


---

## 五、旋钮 3：x 的采样分布

```
Even:           x ~ Uniform(0, 1)
Left-dense:     x ~ Beta(2, 5)
Right-dense:    x ~ Beta(5, 2)
Center-dense:   x ~ Beta(5, 5)
Two clusters:   x ~ 0.5·Uniform(0, 0.3) + 0.5·Uniform(0.7, 1)
Three clusters: 各 1/3 权重, Uniform(0, 0.15) + Uniform(0.4, 0.6) + Uniform(0.85, 1)
```


---

## 六、生成方案

### 主生成：旋钮 1 × 旋钮 2a（shape × SNR）

```
201 函数配置 × 12 SNR 水平 = 2412 个 case

统一设定:
  旋钮 2b = constant spread
  旋钮 3  = Uniform(0,1)
```

### 干扰生成 A：旋钮 2b（heteroscedasticity）

```
代表性函数 (8 个):
  F01 Linear (slope=1), F05 Saturation (k=10), F08 S-curve (k=20),
  F09 Threshold (δ=0.01), F10 Quadratic peak (Inverted U-shape) (a=10, c=0.5),
  F14 Cubic M-type (center, wide spacing), F15 Pure oscillation (低频), Null

× 3 个 SNR 水平: 1, 5, 20
× 3 种新噪声结构: Increasing / Decreasing / Middle-high

= 72 个 case
```

### 干扰生成 B：旋钮 3（sampling distribution）

```
代表性函数（同上 8 个）

× 3 个 SNR 水平: 1, 5, 20
× 5 种采样: Left-dense / Right-dense / Center-dense / Two clusters / Three clusters

= 120 个 case
```

### 生成总量

```
主生成:       2412 case
干扰生成 A:     72 case
干扰生成 B:    120 case
总计:         2604 case
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
