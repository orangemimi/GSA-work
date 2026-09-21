# 散点图构造与标注工作流：三旋钮框架

本文档的目标：系统构造带有已知生成机制和已知标签的 synthetic scatterplots。
本阶段只做**生成 + 标注**，不涉及 metric 计算。


---

## 一、生成模型

任何一个 synthetic scatterplot 由且仅由三个独立旋钮决定：

```
y = f(x) + ε(x)
x ~ sampling distribution
```

| 旋钮 | 控制对象 | 决定的特征 |
|------|---------|-----------|
| 旋钮 1: f(x) | 均值函数的形状和参数 | Direction, Monotonicity, Linearity, Convexity/Concavity, Saturation, Turning Point, S-curve, Threshold |
| 旋钮 2a: σ₀ | 噪声大小 | 与 f(x) 共同决定 Strength (SNR), High Spread |
| 旋钮 2b: σ(x) | 噪声是否随 x 变化 | Changing Spread |
| 旋钮 3: x 分布 | 采样密度和分组结构 | Density (Even/Uneven), Separated Clusters |

**Strength 不是旋钮，是读数。** SNR = Var(f(x)) / Var(ε)，由旋钮 1 和旋钮 2a 共同决定。


---

## 二、旋钮选项空间

### 旋钮 1：f(x) 函数库

按 turning point 数量组织。每个函数类型内部有形状参数连续扫描。

设计原则：每个函数类型 ~20 个形状参数组合，配合 50 级 SNR 扫描，得到 ~1000 case / 函数类型。


#### 0 Turning Point (Monotonic)

**[F01] Linear 族**

```
f(x) = a · x

形状参数 (共 20 个组合):
  slope a = -5, -3, -2, -1, -0.5, -0.3, -0.1,
            0.1, 0.3, 0.5, 1, 2, 3, 5,
            0.01, 0.05, 10, 15, 20, 50

含正负方向、极弱到极强的斜率
a 接近 0 时关系极弱 (配合 SNR 扫描可生成 near-null case)

隐含标签: Linear, Monotonic, No Saturation, No TP, No Threshold, No S-curve
         Direction = sign(a)
```

**[F02] Power-Convex 族**

```
f(x) = xᵖ    (x ∈ [0, 1])

形状参数 (共 20 个组合):
  p = 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.3, 2.5, 3.0, 3.5,
      4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0

p→1 时退化为线性 (过渡带), p 大时 convexity 极强

隐含标签: Nonlinear (p≠1), Monotonic, Convex (f''>0), Positive Direction
         No Saturation, No TP
```

**[F03] Power-Concave 族**

```
f(x) = xᵖ    (x ∈ [0, 1])

形状参数 (共 20 个组合):
  p = 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35,
      0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02

p→1 时退化为线性, p→0 时 concavity 极强 (接近 step-like)

隐含标签: Nonlinear, Monotonic, Concave (f''<0), Positive Direction
         Saturation 取决于末端斜率 (p 很小时末端斜率仍 >0, 不是真正 saturation)
```

**[F04] Saturation 族**

```
f(x) = 1 - exp(-k · x)    (x ∈ [0, 1])

形状参数 (共 20 个组合):
  k = 0.3, 0.5, 0.8, 1, 1.5, 2, 3, 4, 5, 6,
      7, 8, 10, 12, 15, 18, 20, 25, 30, 40

k < 1 时接近线性; k > 10 时强 saturation (末端斜率→0)
这个连续谱可以精确测定 saturation 被"认出"的 k 阈值

隐含标签: Monotonic, Concave, Positive Direction
         Saturation 强度随 k 递增 (用 late_slope/early_slope ratio 量化)
```

**[F05] Log 族**

```
f(x) = log(1 + a · x)    (x ∈ [0, 1])

形状参数 (共 20 个组合):
  a = 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30,
      40, 50, 70, 100, 150, 200, 300, 500, 700, 1000

与 saturation 族的关键区别: log 永远不会真正饱和到斜率=0
a 极大时曲线越来越像 saturation, 但数学上末端斜率 = a/(1+a) · 1/x 仍 >0

隐含标签: Nonlinear, Monotonic, Concave, Positive Direction
         Saturation = No 或 Weak (看具体 a 值下的 late/early slope ratio)
```

**[F06] Exponential 族**

```
f(x) = exp(b · x) - 1    (x ∈ [0, 1])

形状参数 (共 20 个组合):
  b = 0.1, 0.3, 0.5, 0.8, 1, 1.5, 2, 2.5, 3, 3.5,
      4, 4.5, 5, 6, 7, 8, 9, 10, 12, 15

b 小时接近线性; b 大时 convexity 极强, 增长加速

隐含标签: Nonlinear, Monotonic, Convex, Positive Direction
         No Saturation
注: 不同于 Power-Convex 族, exponential 的增长是"越来越快"而不是"幂律加速"
```

**[F07] S-curve 族 (smooth transition between asymptotic levels)**

```
f(x) = 1 / (1 + exp(-k · (x - c)))    (x ∈ [0, 1])

形状参数 (共 20 个组合, 来自 k × c 网格):
  steepness k = 3, 5, 8, 12, 20, 30, 50, 80, 120, 200
  transition center c = 0.3, 0.5, 0.7

  取 10k × 2c 的子集 (去掉 c=0.3/0.7 的极端 k 组合, 保留 ~20 个有意义的配置)

  完整列表:
    k=3,c=0.5 | k=5,c=0.3 | k=5,c=0.5 | k=5,c=0.7
    k=8,c=0.3 | k=8,c=0.5 | k=8,c=0.7
    k=12,c=0.3 | k=12,c=0.5 | k=12,c=0.7
    k=20,c=0.3 | k=20,c=0.5 | k=20,c=0.7
    k=30,c=0.5 | k=50,c=0.3 | k=50,c=0.5 | k=50,c=0.7
    k=80,c=0.5 | k=120,c=0.5 | k=200,c=0.5

隐含标签: Nonlinear, Monotonic, Convexity=Mixed (inflection point at c)
         Positive Direction, No TP
关键特征: f'(x) 处处连续, 两端渐近但无真正稳态区间
```

**[F08] Threshold 族 (abrupt state change: 稳态 A → 突变 → 稳态 B)**

```
f(x) = { a,                              x < c - δ       (稳态 A)
        { a + (b-a)/(2δ)·(x - c + δ),    c-δ ≤ x ≤ c+δ   (过渡带)
        { b,                              x > c + δ       (稳态 B)

形状参数 (共 20 个组合, 来自 gap × δ × c 网格):
  state gap (b - a) = 0.5, 1, 2, 5
  transition width δ = 0.005, 0.01, 0.03, 0.05, 0.1
  position c = 0.5  (固定在中心, 减少组合数; 位置效应在干扰实验中测试)

  取 4 gap × 5 δ = 20 个配置

隐含标签: Piecewise, 过渡带外斜率=0, Positive Direction
关键特征: f'(x) 在 c±δ 处不连续, 存在明确的"稳态区间"
ESM 含义: tipping point — 系统从一个 regime 翻转到另一个 regime
与 S-curve 的核心区别: 过渡带外有真正的 f'(x)=0 平坦段
```


#### 1 Turning Point

**[F09] Quadratic ∩ (peak) 族**

```
f(x) = -a · (x - c)² + d    (x ∈ [0, 1], d 使 f 的 range 归一化)

形状参数 (共 20 个组合, 来自 a × c 网格):
  curvature a = 1, 2, 4, 8, 16
  peak position c = 0.2, 0.35, 0.5, 0.65, 0.8

  4a × 5c = 20 个配置 (去掉 a=16,c=0.2 等极端组合, 补回其他)

隐含标签: Nonlinear, Non-monotonic, TP_count=1, TP_type=peak
         Direction 取决于 c 位置 (c<0.5 时大部分在下降段 → Negative; c>0.5 → Positive)
```

**[F10] Quadratic ∪ (valley) 族**

```
f(x) = a · (x - c)² + d    (x ∈ [0, 1])

形状参数 (共 20 个组合):
  与 F09 相同的 a × c 网格

隐含标签: Nonlinear, Non-monotonic, TP_count=1, TP_type=valley
```

**[F11] Spike 族 (narrow peak)**

```
        { s_up · x,                                          x < c - w/2
f(x) =  { s_up·(c-w/2) + h·(1 - |x-c|/(w/2)),              c-w/2 ≤ x ≤ c+w/2
        { s_up·(c-w/2) + s_down·(x - c - w/2),              x > c + w/2

简化: baseline 近乎平坦 (s_up, s_down ≈ 0), peak 高度 h, 宽度 w

形状参数 (共 20 个组合):
  peak width w = 0.01, 0.02, 0.05, 0.1, 0.2
  peak height h = 0.5, 1, 2, 5
  position c = 0.5  (固定)

  5w × 4h = 20 个配置

隐含标签: TP_count=1, TP_type=peak
         width 极窄时大部分数据看不到 peak → global metrics 接近 null
```

**[F12] L-shaped 族 (narrow valley)**

```
Spike 的倒置版, narrow valley

形状参数 (共 20 个组合):
  与 F11 相同结构, 取负

隐含标签: TP_count=1, TP_type=valley
```


#### 2 Turning Points

**[F13] Cubic M 型 (peak → valley) 族**

```
f(x) = a · (x - r₁)(x - r₂)(x - r₃)    (x ∈ [0, 1])

通过 roots 控制两个 turning point 的位置和间距

形状参数 (共 20 个组合):
  roots 间距 (r₃-r₁) = 0.3, 0.5, 0.7, 0.9
  roots 对称性 (中心偏移) = -0.1, 0, +0.1
  amplitude a: 调整使 f 的 range 归一化
  → ~12 个 roots 配置 + 对每个配置做 1-2 种 amplitude 变体 ≈ 20

隐含标签: Nonlinear, Non-monotonic, TP_count=2, TP_pattern=peak-valley
```

**[F14] Cubic W 型 (valley → peak) 族**

```
f(x) = -a · (x - r₁)(x - r₂)(x - r₃)    (取负)

形状参数: 与 F13 相同, sign(a) 翻转

共 20 个配置
隐含标签: TP_count=2, TP_pattern=valley-peak
```

**[F15] Double Gaussian 族**

```
f(x) = A₁·exp(-(x-μ₁)²/(2σ₁²)) + A₂·exp(-(x-μ₂)²/(2σ₂²))
(valley 型取 -A)

形状参数 (共 20 个组合):
  peak 间距 |μ₂-μ₁| = 0.2, 0.3, 0.4, 0.6
  高度比 A₁:A₂ = 1:1, 2:1, 1:2
  宽度 (σ₁=σ₂) = 0.05, 0.1

  4 间距 × 3 高度比 × 2 宽度 = 24, 从中选 20 个代表性组合

隐含标签: Nonlinear, Non-monotonic, TP_count=2
         可独立控制每个 peak 的位置、宽度、高度
```


#### Multiple Turning Points

**[F16] Pure Oscillation 族**

```
f(x) = A · sin(ω · x)    (x ∈ [0, 1])

形状参数 (共 20 个组合):
  frequency ω = 2π, 3π, 4π, 5π, 6π, 8π, 10π, 12π, 16π, 20π
  amplitude A = 0.3, 1.0

  10ω × 2A = 20 个配置

隐含标签: Nonlinear, Non-monotonic, Direction=None (无整体趋势)
         TP_count = ω/(π) - 1 (在 [0,1] 上)
         TP_pattern = alternating, TP_trend = none
```

**[F17] Oscillation + Trend 族**

```
f(x) = b · x + A · sin(ω · x)    (x ∈ [0, 1])

形状参数 (共 20 个组合):
  frequency ω = 2π, 4π, 6π, 8π, 12π
  trend slope b = -1, -0.3, 0.3, 1
  → 5ω × 4b = 20 个配置
  amplitude A 根据 |b| 调整, 使振荡幅度和趋势幅度的比值固定 (如 A = 0.3·|b|/1)

隐含标签: Non-monotonic, TP_count=multiple
         TP_trend = positive (b>0) / negative (b<0)
         Direction = sign(b)
关键: trend 和 oscillation 的相对强度决定整体 pattern
```

**[F18] Damped Oscillation 族**

```
f(x) = A · exp(-λ · x) · sin(ω · x)    (x ∈ [0, 1])

形状参数 (共 20 个组合):
  frequency ω = 4π, 6π, 8π, 12π, 16π
  damping rate λ = 0.5, 1, 3, 5

  5ω × 4λ = 20 个配置

隐含标签: Non-monotonic, TP_count=multiple, TP_trend=damped
ESM 含义: 参数增大后系统逐渐趋于稳定
```

**[F19] Growing Oscillation 族**

```
f(x) = A · exp(λ · x) · sin(ω · x)    (x ∈ [0, 1])

形状参数 (共 20 个组合):
  frequency ω = 4π, 6π, 8π, 12π, 16π
  growth rate λ = 0.3, 0.5, 1, 2

  5ω × 4λ = 20 个配置

隐含标签: Non-monotonic, TP_count=multiple, TP_trend=growing
ESM 含义: 参数增大后系统越来越不稳定
```

**[F20] Varying Frequency 族**

```
f(x) = A · sin(ω · x · (1 + α·x))    (x ∈ [0, 1])

形状参数 (共 20 个组合):
  base frequency ω = 3π, 5π, 7π, 10π, 14π
  frequency modulation α = 0.5, 1, 2, 3

  5ω × 4α = 20 个配置

隐含标签: Non-monotonic, TP_count=multiple, 频率随 x 递增
参考 Reshef Table S3 的 Varying Freq Sine/Cosine 系列
```


#### No Relationship

**[F00] Random (null baseline)**

```
f(x) = constant (如 0.5)

等价于 y = 0.5 + ε, y 与 x 完全无关
没有形状参数可扫, 但仍需 50 级 SNR 扫描 (此时 SNR 的含义退化为纯噪声大小)
实际操作: 生成 20 组不同的 constant + noise σ 组合, 覆盖不同的 y 分布宽度

共 20 个配置

所有 shape 标签的理论预期 = None / No / 0
用于建立每个 metric 的 null distribution
```


#### 旋钮 1 汇总

```
F00  Random                    20 个配置
F01  Linear                    20
F02  Power-Convex              20
F03  Power-Concave             20
F04  Saturation                20
F05  Log                       20
F06  Exponential               20
F07  S-curve                   20
F08  Threshold                 20
F09  Quadratic ∩               20
F10  Quadratic ∪               20
F11  Spike (narrow peak)       20
F12  L-shaped (narrow valley)  20
F13  Cubic M (peak→valley)     20
F14  Cubic W (valley→peak)     20
F15  Double Gaussian           20
F16  Pure Oscillation          20
F17  Oscillation + Trend       20
F18  Damped Oscillation        20
F19  Growing Oscillation       20
F20  Varying Frequency         20
──────────────────────────────────
共 21 个函数类型 × 20 个形状配置 = 420 个函数配置
```


---

### 旋钮 2：ε(x) 噪声

#### 旋钮 2a：噪声大小 — SNR 扫描

每个函数配置 × 50 级 SNR，构成主实验的核心。

```
SNR 定义:
  SNR = Var(f(x)) / Var(ε)

SNR 取值 (共 50 级, log-uniform 分布):
  0.05, 0.08, 0.1, 0.13, 0.17, 0.2, 0.25, 0.3, 0.4, 0.5,
  0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0,
  3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10, 12, 15, 18,
  20, 25, 30, 35, 40, 50, 60, 80, 100, 130,
  170, 200, 300, 500, 700, 1000, 2000, 5000, 10000, ∞(无噪声)

实现方式:
  给定 f(x), 先在 x ~ Uniform(0,1) 上 (n=10000) 计算 Var_f = Var(f(x))
  σ_ε = sqrt(Var_f / SNR)
  ε ~ N(0, σ_ε), constant across x
  SNR = ∞ 时 σ_ε = 0 (无噪声)

低 SNR 端 (0.05-1): 噪声主导, 大部分 metric 应接近 null baseline
中 SNR 端 (1-20): 信号逐渐显现, 是 metric 分辨力的关键区间
高 SNR 端 (20-∞): 信号主导, metric 应稳定反映形状特征
```

#### 旋钮 2b：噪声结构 (heteroscedasticity)

仅在干扰实验中使用，不进入主实验。

```
Constant spread:       σ(x) = σ₀                              (baseline, 主实验默认)
Increasing spread:     σ(x) = σ₀ · (0.2 + 1.6·x)             (末端噪声大)
Decreasing spread:     σ(x) = σ₀ · (1.8 - 1.6·x)             (前端噪声大)
Middle-high spread:    σ(x) = σ₀ · (0.3 + 1.4·exp(-(x-0.5)²/0.02))

设计原则: E[σ(x)] ≈ σ₀, 保证总噪声量级不变, 只改变空间分布
```


---

### 旋钮 3：x 的采样分布

仅在干扰实验中使用，主实验统一用 Uniform。

```
Even:           x ~ Uniform(0, 1)                                        (主实验默认)
Left-dense:     x ~ Beta(2, 5)
Right-dense:    x ~ Beta(5, 2)
Center-dense:   x ~ Beta(5, 5)
Two clusters:   x ~ 0.5·Uniform(0, 0.3) + 0.5·Uniform(0.7, 1)
Three clusters: x ~ 1/3·Uniform(0, 0.15) + 1/3·Uniform(0.4, 0.6) + 1/3·Uniform(0.85, 1)
```


---

## 三、标签体系和推导规则

每个 case 的标签从生成参数**推导**，不做主观判断。

```
Direction:
  Positive    := f(x_max) > f(x_min)    (整体上升)
  Negative    := f(x_max) < f(x_min)    (整体下降)
  None        := |f(x_max) - f(x_min)| / range(f) < 0.05   (如纯振荡)

Monotonicity:
  Monotonic   := f'(x) 在定义域上不变号
  Non-mono    := f'(x) 至少变号一次

Linearity:
  Linear      := max|f(x) - linear_fit(x)| / range(f) < 0.05
  Nonlinear   := 否则

Convexity (仅对 monotonic 有意义):
  Convex      := f''(x) > 0 throughout
  Concave     := f''(x) < 0 throughout
  Mixed       := f''(x) 变号

Saturation:
  判据: |f'(x_late)| / |f'(x_early)| < 0.1
  x_early = x range 前 10%, x_late = x range 后 10%
  区别于 Concave: concave 的末端斜率下降但不趋近零

Turning Points:
  TP_count    := f'(x) = 0 的解的数量 (0, 1, 2, multiple)
  TP_types    := 每个极值点: peak (f'' < 0) 或 valley (f'' > 0)
  TP_pattern  := 排列 (peak / valley / peak-valley / valley-peak / alternating)
  TP_trend    := 仅 multiple TP: none / positive / negative / damped / growing

S-curve vs Threshold:
  S-curve     := smooth sigmoid, f'(x) 处处连续, 无真正稳态区间
  Threshold   := piecewise, 过渡带外 f'(x)=0 (真正稳态), 过渡带内斜率突变
  判别关键: 过渡带外是否存在 f'(x)≈0 的平坦段

Strength (推导值, 非旋钮):
  Strong      := SNR > 10
  Medium      := 1 < SNR ≤ 10
  Weak        := SNR ≤ 1
  Very Weak   := SNR < 0.3

High Spread:
  Yes         := E[σ(x)] / range(f) > 0.3
  No          := 否则

Changing Spread:
  Yes         := σ(x) 是 x 的非常数函数
  No          := σ(x) = constant

Density:
  Even        := x ~ Uniform
  Uneven      := 否则

Clusters:
  Yes         := x 的分布有 ≥2 个不重叠的 mode
  No          := 否则
```


---

## 四、实验设计

### 主实验：旋钮 1 × 旋钮 2a (shape × SNR)

全面交叉。每个函数类型 ~1000 case。

```
420 函数配置 × 50 SNR 水平 = 21,000 个 case

每个 case 固定:
  旋钮 2b = constant spread
  旋钮 3  = Uniform(0, 1)
  n = 500
  R = 100 次随机实现

每个函数类型 (如 Saturation 族):
  20 形状配置 × 50 SNR = 1000 case
```

### 干扰实验 A：旋钮 2b (heteroscedasticity)

选择性交叉，测试 Changing Spread 对标签可识别性的干扰。

```
代表性函数 (8 类, 每类取 1 个典型配置):
  F01 Linear (a=1)
  F04 Saturation (k=10)
  F07 S-curve (k=20, c=0.5)
  F08 Threshold (gap=1, δ=0.01, c=0.5)
  F09 Quadratic ∩ (a=4, c=0.5)
  F13 Cubic M (典型配置)
  F17 Oscillation+Trend (典型配置)
  F00 Random

× 10 个 SNR 水平: 0.5, 1, 2, 5, 10, 20, 50, 100, 500, ∞
× 3 种噪声结构: Increasing / Decreasing / Middle-high spread

= 8 × 10 × 3 = 240 个 case
对照组: 同一 shape × SNR 在 constant spread 下的值 (已在主实验中)
```

### 干扰实验 B：旋钮 3 (sampling distribution)

选择性交叉，测试采样不均匀和 cluster 对标签可识别性的干扰。

```
代表性函数 (同上 8 类)
× 10 个 SNR 水平: 同上
× 5 种采样: Left-dense / Right-dense / Center-dense / Two clusters / Three clusters

= 8 × 10 × 5 = 400 个 case
对照组: 同一 shape × SNR 在 Uniform 下的值 (已在主实验中)
```

### 实验总量

```
主实验:          21,000 case
干扰实验 A:         240 case
干扰实验 B:         400 case
──────────────────────────────
总计:            21,640 case

每个 case × R=100 次实现 = 2,164,000 个散点图
每个散点图 n=500 个点
```


---

## 五、固定实验参数

```
样本量:      n = 500 per realization
x 归一化:    所有 case 的 x ∈ [0, 1]
y 归一化:    生成后不归一化, 保留 raw range (metric 端按需归一化)
重复次数:    R = 100 per case
随机种子:    seed = case_id × 1000 + r   (r = 1, ..., 100)
```


---

## 六、每个 case 的输出

对每个 case，保存以下信息：

```
case_metadata:
  case_id:          唯一标识 (如 "F04_k10_SNR5.0")
  function_type:    F00-F20
  function_formula: 具体公式字符串
  shape_params:     {k: 10}  或  {a: 4, c: 0.5}  等
  snr:              5.0
  sigma_epsilon:    计算得到的 σ_ε 值
  noise_structure:  "constant" / "increasing" / ...
  x_distribution:   "uniform" / "beta_2_5" / ...
  n:                500
  R:                100

ground_truth_labels:
  direction:        "Positive"
  monotonicity:     "Monotonic"
  linearity:        "Nonlinear"
  convexity:        "Concave"
  saturation:       "Strong"   (附 late_slope/early_slope ratio)
  tp_count:         0
  tp_types:         null
  tp_pattern:       null
  tp_trend:         null
  s_curve:          false
  threshold:        false
  strength:         "Medium"   (SNR=5.0)
  high_spread:      false
  changing_spread:  false
  density:          "Even"
  clusters:         false

data:
  realizations:     R 个 (x, y) 数组, 每个 shape (500, 2)
  f_x_clean:        无噪声的 f(x) 值 (500,), 用于验证
```
