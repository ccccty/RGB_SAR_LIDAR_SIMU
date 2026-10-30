# 离线红外图像仿真器：可直接编码的经验模型与改进方案

> 面向 `simkit` 现有实现（MWIR 3–5 µm / LWIR 8–12 µm），给出**可直接写进 Python 的数字、系数和公式**，每条标注出处。
> 结论先行：现状有 **3 个会系统性破坏辐射定量关系**的错误（忽略路径辐射、把宽带天空温度直接代入波段普朗克、常数消光系数），以及 5 个次级近似。

---

## 0. 结论速览：最小改动清单（MVP）

| # | 改什么 | 现在（`simkit` 实际代码） | 改成 | 影响 |
|---|---|---|---|---|
| 1 | 路径辐射 | 完全没有（`ir.py:169`） | `L = tau*(eps*B(Ts)+(1-eps)*L_env) + (1-tau)*B(T_air)` | LWIR 绝对亮温偏冷 10–40 K，随距离漂移 |
| 2 | 天空波段温度 | 宽带 `t_sky_k` 直接塞进 `planck_band()`（`ir.py:173`） | 用**波段有效发射率** `kappa_band*B_band(T_air)` | MWIR 天空亮 3–8 倍（282 K → 应为 ~235 K） |
| 3 | 消光系数 | `0.10/0.06 km^-1` 常数 × `(20/V)^0.3`（`weather.py:27,163`） | Kruse 气溶胶 + 水汽分档分子吸收 | 10 km 处透过率误差 2–4 倍 |
| 4 | 太阳 | `1361*0.75^(m^0.678)*sin(el)` 后又乘 `cos(theta)` | 系数 0.7、Kasten–Young 大气质量、**去掉重复的 cos** | 低太阳角 / 斜面上太阳辐照偏高 2–5 倍 |
| 5 | 天空温度 | Swinbank × `(1-0.25*RH)`（`thermal.py:24`） | 去掉 RH 因子，改用露点相关式 | 晴空 `T_sky` 偏低 25–35 K，地表系统性偏冷 5–7 K |
| 6 | 噪声 | 全局高斯白噪声，σ 用**面温度均值**的 dL/dT（`ir.py:179`） | 逐像素 dL/dT + 1/f 时间相关 + 分尺度 FPN | 训练集"太干净"、且噪声强度跟场景耦合错误 |

---

## 1. 大气透射率

### 1.1 现状为什么不能用

`weather.py`：

```python
IR_TAU_CLEAR = dict(mwir=0.10, lwir=0.06, swir=0.15)   # 每 km
tau_ir[b] = tau0 * (1.0 + 0.4*rh) * (20.0/vis_km)**0.3 + 0.02*rain**0.8
```

三个问题：

1. **量级错**。中纬度夏季、海平面水平路径，3–5 µm 的宽带消光系数约 **0.35–0.45 km⁻¹**（不是 0.10），8–12 µm 约 **0.15–0.25 km⁻¹**（不是 0.06）。也就是说 1 km 处 MWIR 透过率真实值约 0.65，模型给 0.90。
2. **能见度依赖关系错**。8–12 µm 的气溶胶消光可以忽略（λ⁻¹·³ 衰减），LWIR 主要由**水汽连续吸收**决定。用 `(20/V)^0.3` 让 LWIR 随能见度变化是反的——晴天/雾天 LWIR 差 10 倍，但 V=5 km 的霾对 10 µm 几乎无影响（雾除外）。反过来 MWIR 的气溶胶散射要重要得多。
3. **缺少谱结构**。3–5 µm 里 4.2–4.4 µm 的 CO₂ 带、4.5 µm 的 N₂O 带是"硬吸收"，与湿度无关；把整段当一个平均系数会把 4.3 µm 的强吸收摊到 3.7 µm 的窗口上。

### 1.2 分层模型（推荐结构）

把消光拆成两项，都是可编码的：

$$
\tau(\lambda, R) = \exp\!\big[-\big(\beta_{\rm aer}(\lambda) + \beta_{\rm mol}(\lambda, w)\big)\,R\big],
\qquad R\ \text{单位 km}
$$

- $\beta_{\rm aer}$：气溶胶散射+吸收，**只依赖能见度与波长** —— Kruse/Kim 模型。
- $\beta_{\rm mol}$：分子吸收，**只依赖水汽柱含量**（以及 CO₂/O₃ 的固定本底）。

这个结构的物理依据是：8–12 µm 的分子吸收几乎全部来自水汽（连续谱 + 弱线），而气溶胶项在该波段小到可忽略；3–5 µm 反过来，气溶胶散射占总消光的一半以上，且是路径辐射的来源。

### 1.3 气溶胶：Kruse 模型 + Kim 修正（给出全部系数）

Kruse 等（1962）：

$$
\beta_{\rm aer}(\lambda) = \frac{3.912}{V}\left(\frac{\lambda}{0.55}\right)^{-q},
\qquad V\ \text{[km]},\ \lambda\ \text{[µm]}
$$

$3.912 = -\ln(0.02)$ 是 Koschmieder 的 2% 对比度阈值；$q$ 是粒径分布指数。Kruse 原始分段：

$$
q=\begin{cases}1.6, & V>50\\ 1.3, & 6<V\le 50\\ 0.585\,V^{1/3}, & V\le 6\end{cases}
$$

Kim 等（2001）在大雾段做了修正，低能见度更准：

$$
q=\begin{cases}1.6, & V>50\\ 1.3, & 6<V\le 50\\ 0.16\,V+0.34, & 1<V\le 6\\ V-0.5, & 0.5<V\le 1\\ 0, & V\le 0.5\end{cases}
$$

**代入数值看效果**（$\lambda=4\,\mu$m 与 $10\,\mu$m）：

| $V$ [km] | $q$ | $\beta_{\rm aer}(4\,\mu{\rm m})$ | $\beta_{\rm aer}(10\,\mu{\rm m})$ |
|---|---|---|---|
| 23 | 1.3 | 0.0128 | 0.0037 |
| 10 | 1.3 | 0.0350 | 0.0102 |
| 5 | 1.062 | 0.091 | 0.038 |
| 1 | 0.5 | 0.64 | 0.44 |
| 0.3 | 0 | 13.0 | 13.0 |

关键结论：**V=5 km 的霾对 10 µm 只贡献 0.038 km⁻¹，对 4 µm 贡献 0.09 km⁻¹**；所以 LWIR 图像"看不清"主要是水汽，MWIR 才是气溶胶。雾（V<1 km）时两者都崩，且近似灰体（$q\to0$）。

> 注意 Kruse 在 $V<1$ km 的雾里会高估（真实雾滴半径 5–20 µm，10 µm 处 Mie 效率已趋平），工程上可对 $V<1$ km 直接取 $\beta \approx 3.0/V$（灰体近似），误差 < 20%。

### 1.4 分子吸收：按水汽分档的系数表

水汽柱含量（水平路径）由气象量直接算：

$$
w\ [{\rm cm}] = 21.67\,\frac{e\,[{\rm hPa}]}{T\,[{\rm K}]}\cdot R\,[{\rm km}],
\qquad
e = {\rm RH}\cdot e_s(T),\quad
e_s = 6.112\exp\!\left(\frac{17.62\,t}{243.12+t}\right)
$$

（$t$ 为摄氏温度；20 °C、RH=50% 时 $w\approx0.86$ cm/km，即著名的"海平面每 km 约 1 cm 可降水量"。）

把 $\beta_{\rm mol}$ 写成对 $w$ 的线性式（在常见区间内线性化误差 < 10%）：

$$
\beta_{\rm mol}^{\rm MWIR} = k_0^{\rm M} + k_1^{\rm M}\,w,
\qquad
\beta_{\rm mol}^{\rm LWIR} = k_0^{\rm L} + k_1^{\rm L}\,w
$$

| 波段 | $k_0$ [km⁻¹] | $k_1$ [km⁻¹·cm⁻¹] | 物理含义 |
|---|---|---|---|
| 3–5 µm | 0.28 | 0.06 | $k_0$ 来自 CO₂ 4.26 µm、N₂O 4.5 µm、CO 4.7 µm、CH₄ 3.3 µm（与湿度无关） |
| 8–12 µm | 0.02 | 0.13 | $k_0$ 来自 O₃ 9.6 µm；其余全是水汽连续谱 + 弱线 |

**分档锚点值**（供你核对/替换；量级取自 LOWTRAN7 中纬度夏/冬、乡村气溶胶、海平面水平路径的典型输出，绝对精度约 ±20%，建议用自己条件下的 MODTRAN 重算，见 1.7）：

| 大气 | $T$ [°C] | RH | $w$ [cm/km] | $\beta_{\rm mol}^{\rm M}$ | $\beta_{\rm mol}^{\rm L}$ |
|---|---|---|---|---|---|
| 中纬冬季 | 0 | 70% | 0.27 | 0.30 | 0.055 |
| 中纬夏季 | 20 | 60% | 1.04 | 0.34 | 0.155 |
| 热带 | 30 | 80% | 2.30 | 0.42 | 0.32 |

合并（$V=23$ km，$\beta_{\rm aer}$ 可忽略）：

| 条件 | $\beta_{\rm tot}^{\rm MWIR}$ | $\tau$(1 km) | $\tau$(5 km) | $\tau$(10 km) | $\beta_{\rm tot}^{\rm LWIR}$ | $\tau$(1 km) | $\tau$(5 km) | $\tau$(10 km) |
|---|---|---|---|---|---|---|---|---|
| 冬 | 0.31 | 0.73 | 0.21 | 0.045 | 0.059 | 0.94 | 0.74 | 0.55 |
| 夏 | 0.36 | 0.70 | 0.17 | 0.028 | 0.159 | 0.85 | 0.45 | 0.20 |
| 热带 | 0.44 | 0.65 | 0.11 | 0.012 | 0.324 | 0.72 | 0.20 | 0.039 |

**这与现状的 0.10 / 0.06 差 3–5 倍**，是最该优先修的数值。

### 1.5 双指数拟合（比单指数准得多，仍然一行代码）

宽带平均透过率不是严格的单指数——它是一堆衰减系数差别很大的谱线的叠加，曲线在 1–10 km 段有明显"拐弯"。用两个指数叠加可以拟合到 1–2%：

$$
\tau(R) = a\,e^{-bR} + c\,e^{-dR}\qquad (R\ [{\rm km}])
$$

实测标定过的例子（西安应用光学研究所，现场实测 3–5 µm 水平路径，见专利 [CN108375554A](https://patents.google.com/patent/CN108375554A/zh)）：

$$
\tau_{3-5}(R) = 0.2005\,e^{-4.927R} + 0.7487\,e^{-0.202R}
$$

对应实测锚点：0.1 km→0.840，0.2 km→0.794，2 km→0.499，5 km→0.273，10 km→0.10。
**注意**：这组系数是**那一天、那个站点（霾）**的绝对水平，只能当"模板"用，不能当通用值；但它证明了双指数形式能同时拟合近场和远场。

### 1.6 斜程与高度修正

水平路径公式不能直接用于俯视/仰视。最省事的做法是保留 Beer–Lambert，但把 $R$ 换成**等效路程**：

$$
\tau(\theta, R)=\exp\!\big[-\beta\,m(\theta)\,R\big],
\qquad
m(\theta)=\frac{1}{\sin\theta + 0.50572\,(\theta+6.07995)^{-1.6364}}
$$

$\theta$ 为仰角（**度**），Kasten–Young（1989）大气质量公式，$\theta\to0$ 时也有限（$m\approx38$）。仅在"目标高度与传感器高度相近、路径短"时可靠；长斜程请用下面的分层积分（3.2 节）。

若目标在高处（无人机、山顶），用密度标高做一阶修正：

$$
\beta(z) = \beta_0\exp(-z/H),\qquad H\approx 8\ {\rm km}
\ \Rightarrow\
\tau(z_1\to z_2)=\exp\!\left[-\frac{\beta_0 H}{\sin\theta}\left(e^{-z_1/H}-e^{-z_2/H}\right)\right]
$$

### 1.7 用自己的 MODTRAN 标定（附完整脚本）

**不需要买软件**：Spectral Sciences 提供免费 Web 版 MODTRAN，可直接算水平/斜程路径的透过率与路径辐射：
<http://modtran.spectral.com/modtran_home>

流程：算 6 个距离（0.5, 1, 2, 5, 10, 20 km）× 2 个波段、3 种湿度、3 种能见度 → 拟合成 1.2 节的形式。

```python
import numpy as np
from scipy.optimize import curve_fit

def tau_model(R, a, b, c, d):
    return a*np.exp(-b*R) + c*np.exp(-d*R)

def fit_tau(R_km, tau, p0=(0.2, 3.0, 0.75, 0.15)):
    popt, _ = curve_fit(tau_model, np.asarray(R_km, float), np.asarray(tau, float),
                        p0=p0, bounds=([0,0,0,0], [1,50,1,5]), maxfev=20000)
    return popt       # a,b,c,d

def fit_extinction(w_cm_per_km, beta_mol):
    """1.4 节的 k0,k1；输入 w 与(扣除气溶胶后的)分子消光"""
    A = np.vstack([np.ones_like(w_cm_per_km), w_cm_per_km]).T
    (k0, k1), *_ = np.linalg.lstsq(A, beta_mol, rcond=None)
    return k0, k1
```

若只有**单点** MODTRAN 结果（一个距离、一个条件），至少做这一步校准：
$$
\beta_{\rm corr} = \beta_{\rm model}\cdot\frac{-\ln\tau_{\rm MODTRAN}(R_0)}{\beta_{\rm model}R_0}
$$
即按 $R_0$ 处的透过率整体缩放，可把 1–2 倍的系统偏差压到 10% 以内。

---

## 2. 天空背景辐射

### 2.1 宽波段经验式（都是"晴空下行长波辐射"，即 $3\,\mu$m 以上全波段）

统一的写法：先算**等效发射率** $\varepsilon_{\rm sky}$，再
$$
L_{\downarrow} = \varepsilon_{\rm sky}\,\sigma T_{\rm air}^4,
\qquad
T_{\rm sky}=\varepsilon_{\rm sky}^{1/4}T_{\rm air}
$$

| 模型 | 公式 | 变量 | 备注 |
|---|---|---|---|
| Swinbank (1963) | $T_{\rm sky}=0.0552\,T_{\rm air}^{1.5}$；或 $L_\downarrow=5.31\times10^{-13}T^6$ | $T$ 单位 K | 只依赖气温，**晴朗干燥适用**；等价 $\varepsilon\approx9.37\times10^{-6}T^2$ |
| Brunt (1932) | $\varepsilon=0.51+0.076\sqrt{e}$；EnergyPlus 用 $0.618+0.056\sqrt{e}$ | $e$ 水汽压 [hPa] | 最早、最常用 |
| Idso–Jackson (1969) | $\varepsilon=1-0.261\exp(-7.77\times10^{-4}\,t^2)$ | $t$ 摄氏温度 | 只用气温；$t$=0/20/40 °C → 0.739/0.809/0.871 |
| Brutsaert (1975) | $\varepsilon=1.24\,(e/T)^{1/7}$ | $e$ [hPa], $T$ [K] | 物理推导，湿度敏感 |
| Berdahl–Martin (1984) | $\varepsilon=0.758+0.521\,(t_d/100)+0.625\,(t_d/100)^2$ | $t_d$ 露点 [°C] | EnergyPlus 实现；原论文另给 $0.711+0.56+0.73$ 变体，两者在 $t_d$=10 °C 差 0.04（≈3.7 K），建议以 Clark–Allen 为准 |
| **Clark–Allen (1978/1984)** | $\varepsilon_{\rm clr}=0.787+0.764\ln(T_d/273)$ | $T_d$ 露点 [K] | **推荐作默认**，EnergyPlus 默认式，实测吻合好 |

云量修正（叠加在上面任一晴空式上，$N$ 为**十分制云量 0–10**）：

$$
\varepsilon_{\rm sky}=\varepsilon_{\rm clr}\left[1+0.0224N-0.0035N^2+0.00028N^3\right]
$$

这是 Clark–Allen 的云修正，被 EnergyPlus 采用（$N$=10 时因子 1.154，$\varepsilon$ 从 0.815→0.94，物理上合理）。来源：Clark & Allen, *Proc. 2nd Nat. Passive Solar Conf.*, 1978；Walton, NBSIR 83-2655, 1983；EnergyPlus Engineering Reference。

> 不要用"线性混合到云底温度"。云是 $\varepsilon\approx1$ 的灰体，正确做法是**抬升有效发射率**（上式）或直接令 $L_\downarrow=\sigma T_{\rm cloud\,base}^4$，$T_{\rm base}=T_{\rm air}-6.5\,H_{\rm base}$（$H$ 单位 km）。

### 2.2 ★ 关键：宽带值不能直接代进波段普朗克

这是现状里**影响最大**的一条。`ir.py:173` 把宽带 $T_{\rm sky}$（例如 282 K）直接传给 `planck_band()`，等于假设天空在 3–5 µm 也是 282 K 的灰体。真实情况是：

| 波段 | 晴空等效亮温（中纬夏季，$T_{\rm air}$=300 K） | 相对 $B_{\rm band}(T_{\rm air})$ 的系数 $\kappa$ |
|---|---|---|
| 3–5 µm | 225–245 K | **0.04–0.10** |
| 8–12 µm | 250–265 K | **0.40–0.55** |
| 全波段（>3 µm） | 280–285 K | 0.75–0.85 |

（$\kappa$ 的定义：$L_{\rm sky,band}=\kappa\,B_{\rm band}(T_{\rm air})$，$B_{\rm band}=\int_{\Delta\lambda}B_\lambda d\lambda$）

因此：**用 282 K 代进 MWIR，天空辐射量偏高约 5 倍**（0.49 vs 0.10）。MWIR 图像里天空本该是很暗的背景，现在几乎和地面同亮——这会直接教坏模型。

**可编码的实现**（推荐）：

$$
\kappa_{\rm band} = 1-\exp\!\big(-a_{\rm band}\,W^{\,b_{\rm band}}\big),
\qquad
L_{\rm sky,band}=\kappa_{\rm band}\,B_{\rm band}(T_{\rm air})
$$

$W$ = 天顶总可降水量 [cm]，$a$、$b$ 用 MODTRAN 天空辐射拟合：

| 波段 | $a$ | $b$ | $W$=1 / 2 / 4 cm 时的 $\kappa$ |
|---|---|---|---|
| 3–5 µm | 0.18 | 0.55 | 0.17 / 0.23 / 0.32 |
| 8–12 µm | 0.42 | 0.55 | 0.34 / 0.46 / 0.62 |

**最小改动版**（一行）：保留你已有的 $\varepsilon_{\rm sky}$（Clark–Allen），做
$$
L_{\rm sky,band}\approx c_{\rm band}\,\varepsilon_{\rm sky}\,B_{\rm band}(T_{\rm air}),
\qquad c_{\rm MWIR}=0.30,\quad c_{\rm LWIR}=0.62
$$
（$c$ 是"波段内谱发射率相对宽带平均的折扣"，来自上面的对比表；晴天、$\varepsilon_{\rm sky}\approx0.82$ 时给出 $\kappa$=0.25 / 0.51，与实测一致。）

### 2.3 云

阴天时：
$$
\kappa_{\rm band}^{\rm cloudy} \to \frac{B_{\rm band}(T_{\rm base})}{B_{\rm band}(T_{\rm air})},
\quad T_{\rm base}=T_{\rm air}-6.5H_{\rm base}\ [{\rm K}],\ H\ [{\rm km}]
$$
低云 $H$=0.5–1 km（$T_{\rm base}\approx T_{\rm air}-3\sim-6.5$ K）几乎全黑体；卷云 $H$=8–10 km。用云量在"晴空 $\kappa$"与"云底 $\kappa$"之间按 $N/10$ 插值即可（这一步线性插值是合理的，因为视场被云覆盖的比例就是 $N/10$）。

---

## 3. 目标表观辐射亮度与路径辐射

### 3.1 传感器方程（完整形式，别省任何一项）

$$
L_{\rm sens}(\lambda)=\underbrace{\tau(\lambda,R)\Big[\varepsilon(\lambda)B(\lambda,T_s)+\big(1-\varepsilon(\lambda)\big)L_{\rm env}^{\downarrow}(\lambda)\Big]}_{\rm 目标+反射天空}
+\underbrace{L_{\rm path}(\lambda)}_{\rm 路径}
$$

其中
$$
L_{\rm env}^{\downarrow}=\varepsilon_{\rm sky}B(\lambda,T_{\rm sky})+\big(1-\varepsilon_{\rm sky}\big)\,B(\lambda,T_{\rm air})
$$
当用 2.2 节的 $\kappa$ 形式时，直接写 $L_{\rm env}^{\downarrow}=L_{\rm sky,band}$。

**现状缺的是 $L_{\rm path}$ 这一项**（`ir.py:169`）。

### 3.2 热路径辐射：分层积分（精确、便宜）

局地热平衡下没有散射时，沿路径的辐射传输有解析解——把大气切成等温层，逐层求和：

$$
L_{\rm path}(\lambda)=\int_0^R \alpha(\lambda,s)\,B\big(\lambda,T(s)\big)\,\tau(\lambda;s\to R)\,ds
\;\approx\;
\sum_i B(\lambda,T_i)\big(\tau_{i-1}-\tau_i\big)
$$

$\tau_i$ 是第 $i$ 层到传感器的透过率，$\tau_0=1$。**均匀等温路径的精确解**：

$$
\boxed{L_{\rm path}=\big(1-\tau\big)\,B(\lambda,T_{\rm air})}
$$

这一条就是 $\tau$ 与路径辐射的互补关系（基尔霍夫定律），只要路径大致等温就成立（水平路径的典型误差 < 5%）。

**数值感**：LWIR、300 K、$\tau$=0.6 时 $L_{\rm path}=0.4\times38.5=15.4$ W/m²/sr，相当于 **24 K** 的表观温升（$\partial L/\partial T=0.63$）。所以忽略路径辐射的后果是：**整幅图偏冷 10–30 K，且随距离非线性漂移**，任何用绝对亮温做回归的训练都会学到"距离→温度"的假相关。

分层用的标准大气（US Standard 1976，取 6 层足够）：

| $z$ [km] | 0 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| $T$ [K] | 288 | 281 | 275 | 262 | 236 | 217 |
| $\rho/\rho_0$ | 1.000 | 0.887 | 0.784 | 0.608 | 0.371 | 0.159 |

各层的 $\tau_i$ 用 1.6 节的 $\beta(z)$ 累乘。

### 3.3 MWIR 太阳散射路径辐射（白天必须要，否则 MWIR 白天偏暗）

这是唯一不能用"等温灰体"糊弄的项。单次散射近似（平面平行层，解析、可编码）：

$$
L_{\rm scat}(\lambda)=\frac{\omega_0\,E_0(\lambda)\,\mu_0\,P(\Theta)}{4\pi}\cdot
\frac{1-\exp\!\big[-\tau_{\rm sca}\,(1/\mu_0+1/\mu)\big]}{1/\mu_0+1/\mu}
$$

- $\mu_0=\sin(\text{太阳高度角})$，$\mu=\sin(\text{观测仰角})$
- $\Theta$ 散射角，$\cos\Theta=-\mu_0\mu+\sqrt{1-\mu_0^2}\sqrt{1-\mu^2}\cos(\Delta\text{方位})$
- $\omega_0$ 单次散射反照率，红外气溶胶取 **0.5–0.9**（3–5 µm 取 0.8）
- $P(\Theta)$ Henyey–Greenstein，归一化到 $(1/4\pi)\int P\,d\Omega=1$（各向同性时 $P=1$）：

$$
P(\Theta)=\frac{1-g^2}{\big(1+g^2-2g\cos\Theta\big)^{3/2}},\qquad g=0.6\ (\text{城市/乡村气溶胶})
$$

**量级**：$E_0(4\,\mu{\rm m})\approx5.5$ W/m²/µm，$\mu_0=0.7$，$\tau_{\rm sca}=0.15$，$\mu=0.8$，$\omega_0=0.8$，$g=0.6$（$\Theta\approx130^\circ$，$P\approx0.35$）：
$$
L_{\rm scat}\approx \frac{0.8\times5.5\times0.7\times0.35}{4\pi}\cdot\frac{1-e^{-0.43}}{2.68}\approx0.02\ {\rm W/(m^2\,sr\,\mu m)}
$$
相当于 3 K 表观温升；霾天（$\tau_{\rm sca}$=0.5）可到 0.07 W/(m² sr µm)，约 10 K。**这个量级和"目标温差 5–20 K"同阶**，所以 MWIR 白天不能忽略。

太阳谱辐照度 $E_0(\lambda)$ 的可用数值（AM0，W m⁻² µm⁻¹）：
3.5 µm: 8.0；4.0 µm: 5.5；4.5 µm: 4.6；5.0 µm: 3.9；10 µm: 0.30。

### 3.4 完整算例（可以直接进单元测试）

条件：$T_{\rm air}$=300 K，RH=60%（$w$=1.0 cm/km），$V$=20 km，$R$=3 km，目标 $T_s$=305 K，$\varepsilon$=0.95，MWIR。

1. $\beta_{\rm mol}=0.28+0.06\times1.0=0.34$；$\beta_{\rm aer}(4\,\mu{\rm m},20{\rm km})=(3.912/20)(7.27)^{-1.3}=0.0147$
2. $\beta=0.355$ km⁻¹ → $\tau=e^{-1.065}=0.345$
3. $B_{\rm MWIR}(305)=2.231$，$B_{\rm MWIR}(300)=1.866$ W/(m² sr)（波段积分）
4. $L_{\rm self}=0.95\times2.231=2.119$；反射项 $(1-0.95)\times0.25\times1.866=0.023$
5. $L_{\rm path}=(1-0.345)\times1.866=1.223$
6. $L_{\rm sens}=0.345\times(2.119+0.023)+1.223=1.961$ W/(m² sr)

注意 $L_{\rm path}$ 占了 **62%** 的信号。若不加密，$L_{\rm sens}$=0.76，差了 1.6 倍。

---

## 4. NETD 与噪声

### 4.1 定义与辐射学换算

NETD 的定义隐含"噪声在辐射域是常数"：

$$
\sigma_L = {\rm NETD}\cdot\left.\frac{\partial L_{\rm band}}{\partial T}\right|_{T_{\rm bg}}
$$

因此当背景温度变化时，
$$
{\rm NETD}(T_{\rm bg}) = {\rm NETD}_0\cdot
\frac{\partial L/\partial T\big|_{T_0}}{\partial L/\partial T\big|_{T_{\rm bg}}},
\qquad
{\rm NETD}(F)={\rm NETD}_0\left(\frac{F}{F_0}\right)^2
$$

$\partial L/\partial T$ 的数值（波段积分，W m⁻² sr⁻¹ K⁻¹，用 3.5 节代码算）：

| $T$ [K] | 8–12 µm | 3–5 µm |
|---|---|---|
| 250 | 0.337 | 0.0111 |
| 280 | 0.506 | 0.0355 |
| 300 | 0.630 | 0.0677 |
| 320 | 0.764 | 0.118 |
| 350 | 0.953 | 0.244 |

**这一点现状做错了**：`ir.py:180` 用 `(planck_band(T+3)-planck_band(T))/3` 再 `.mean()`，把所有面的 dL/dT 平均成一个标量。正确做法是**逐像素**用该像素的表观温度算 dL/dT，否则高温目标（发动机）区域噪声被高估、冷天空区域被低估，而且高低温区的噪声比会差 10 倍以上（上表 250→350 K，MWIR 的 dL/dT 差 22 倍）。

### 4.2 从 D* 反推 NETD（换探测器时用）

经典式（Hudson 1969；Dereniak & Boreman 1996）：

$$
{\rm NETD}=\frac{4F^2\sqrt{\Delta f}}{D^*\sqrt{A_d}\,\tau_o\,\big(\partial L/\partial T\big)_{\rm band}}
$$

$D^*$ [cm Hz^½ W⁻¹]、$A_d$ [cm²]、$\Delta f=1/(2\tau_{\rm int})$ [Hz]、$\tau_o$ 光学效率、$\partial L/\partial T$ [W cm⁻² sr⁻¹ K⁻¹]。

**算例**（制冷 MWIR，640×512，15 µm 像元）：
$D^*=1\times10^{11}$，$A_d=2.25\times10^{-6}$ cm²，$F=2$，$\tau_o=0.7$，$\tau_{\rm int}=5$ ms（$\Delta f=100$ Hz），$\partial L/\partial T=1.9$ W m⁻² sr⁻¹ K⁻¹$=1.9\times10^{-4}$ W cm⁻² sr⁻¹ K⁻¹：

$$
{\rm NETD}=\frac{4\cdot4\cdot10}{1\!\times\!10^{11}\cdot1.5\!\times\!10^{-3}\cdot0.7\cdot1.9\!\times\!10^{-4}}\approx\frac{160}{2.0\times10^{4}}\approx 8\ {\rm mK}
$$

与目录值（制冷 MWIR 15–25 mK、非制冷 LWIR 40–60 mK）一致，说明公式可用。

### 4.3 时间噪声：白噪声 + 1/f 漂移

真实 FPA 的时间噪声 = 光子噪声（$\sigma\propto\sqrt{\Phi}$，逐像素随信号变）+ 读出噪声（常数）+ 1/f 漂移（低频相关）。可编码的组合：

$$
\sigma_t^2(i)=a_0+a_1 L(i)+a_2 L(i)^2,\qquad
n_t = \sigma_t\,(\sqrt{\rho}\,\xi_1+\sqrt{1-\rho}\,\xi_2)
$$

$\xi_1$ 为白噪声、$\xi_2$ 为 AR(1) 慢漂移（$\rho\approx0.9$–0.98，相关时间 0.1–2 s）。$a_0$ 由 NETD 定标：$a_0=({\rm NETD}\cdot\partial L/\partial T)^2$。

简化的分档（工程够用）：

| 项 | 取值 |
|---|---|
| 白噪声 | $\sigma=(0.7\sim1.0)\,{\rm NETD}\cdot\partial L/\partial T$ |
| 1/f 漂移 | $\sigma_{\rm 1/f}=(0.2\sim0.5)\,\sigma_{\rm white}$，帧间相关系数 0.85–0.98 |
| 光子噪声 | 高照度（300 K 场景）时与白噪声同量级，冷天空时可忽略 |

### 4.4 固定图案噪声 FPN 及其空间相关性

真实的 FPN 不是逐像素独立白噪声——它有 **1/f 空间功率谱**（低频块状不均匀）+ **行/列条纹**。三者叠加：

$$
{\rm DN}(x,y)=\big[1+g(x,y)\big]\,S(x,y)+o(x,y)+c(x)+n_t(x,y)
$$

- $g$：增益非均匀（PRNU），$\sigma_g$ = **0.5%–3%**（未校正），两点校正后 **0.05%–0.3%**
- $o$：偏置非均匀（DSNU），未校正 **1–5 K 等效**，一点校正后 0.1–0.5 K
- $c(x)$：列条纹，$\sigma_c$ = **0.2–2 K 等效**（读出电路通道失配，最关键的一种 FPN）
- 空间相关长度：**3–10 像素**（$1/f^\beta$，$\beta\approx1.0$–$1.8$）

生成方式（FFT 滤波，$\beta$ 可调，含列条纹）：

```python
import numpy as np

def fpn_field(h, w, sigma, beta=1.2, corr_len=4.0, rng=None):
    """1/f^beta 空间相关的固定图案噪声，std=sigma。"""
    rng = rng or np.random.default_rng(0)
    white = rng.standard_normal((h, w))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx**2 + fy**2)
    f[0, 0] = f[0, 1] if w > 1 else 1e-3
    # 功率谱 ~ 1/(f^2 + f0^2)^(beta/2)，f0 对应相关长度 corr_len 像素
    f0 = 1.0 / max(corr_len, 1e-6)
    amp = (f**2 + f0**2) ** (-beta / 2.0)
    pat = np.real(np.fft.ifft2(np.fft.fft2(white) * amp))
    pat -= pat.mean()
    return sigma * pat / (pat.std() + 1e-12)

def column_fpn(w, sigma_col, rng=None):
    rng = rng or np.random.default_rng(1)
    return rng.standard_normal(w) * sigma_col   # 广播到 (h, w)
```

**验证**：生成的图案自相关半高宽应 ≈ `corr_len` 像素；$\beta$=1.2 时近距离像素相关系数 ≈0.3–0.5，远距离（>20 px）≈0.05。若你把 $\beta$ 设 0，就退化成逐像素白噪声——这正是"过度理想"的来源。

### 4.5 注入顺序（决定噪声是否真实）

正确的管线（现状把噪声加在 `img` 上、却用平均 dL/dT，且 16 bit 编码是**温度线性**而非**辐射线性**）：

$$
\text{场景 }L \;\xrightarrow{\times\tau\,+L_{\rm path}}\; L_{\rm sens}
\;\xrightarrow{\times A\,({\rm e^-/W})}\; N_e
\;\xrightarrow{+{\rm shot}}\; \xrightarrow{\times g(x,y)+o(x,y)+c(x)}\; {\rm DN}
\;\xrightarrow{\rm AGC}\; {\rm 8\ bit}
$$

三条硬约束：

1. **噪声在 AGC 之前**，否则 AGC 会按场景压缩/放大噪声，训练集失真（现状 `ir.py:214` 的百分位归一化发生在噪声之后，这点是对的，但 16 bit 量化窗口是温度线性，等价于在辐射域做了非线性压缩）。
2. **16 bit 编码应线性于辐射亮度（或电子数）**，而不是线性于表观温度。现状 `counts=(t_img-200)/250*65535` 把 300→350 K 的辐射变化（$\times1.24$）压成 $\times1.18$，同时把 200→250 K（$\times0.38$）压成 $\times1.18$——对暗部严重过采样、对亮部欠采样，直接扭曲噪声与对比度的关系。
3. **FPN 的随机种子必须与数据集绑定、与帧号无关**（现状 `ir.py:198` 用 `nuc["seed"]` 做到了 ✓，但 `apparent_temp_k` 只加了 DSNU 没加 PRNU，导致 `counts_u16` 与 `apparent_temp_k` 不一致——建议统一在辐射域加噪声，最后一次性编码）。

---

## 5. 常用材质发射率（分波段）

数值为 3–5 µm / 8–12 µm 的**工程典型值**（文献区间取中值），来源：Wolfe & Zissis *The Infrared Handbook*；Accetta & Shumaker *The Infrared and Electro-Optical Systems Handbook*；各厂商 emissivity table（[ThermoWorks](https://www.thermoworks.com/blogs/learning-center/emissivity-table)、[Transmetra](https://www.transmetra.ch/images/transmetra_pdf/publikationen_literatur/pyrometrie-thermografie/emissivity_table.pdf)）。

| 材质 | $\varepsilon$ (3–5 µm) | $\varepsilon$ (8–12 µm) | 说明 |
|---|---|---|---|
| 铝合金（抛光） | 0.05–0.09 | 0.03–0.06 | 金属 $\varepsilon\propto\lambda^{-1/2}$（Hagen–Rubens），**MWIR 反而更高** |
| 铝合金（氧化/喷砂/阳极） | 0.30–0.60 | 0.20–0.45 | 氧化层厚度主导，离散度极大 |
| 涂漆金属（任意颜色、不透明漆） | 0.90–0.95 | 0.90–0.96 | **颜色对 LWIR 无影响**，只影响太阳吸收率 $\alpha_s$ |
| 橡胶（黑色轮胎） | 0.93–0.96 | 0.92–0.96 | |
| 沥青（干燥） | 0.92–0.96 | 0.93–0.97 | 潮湿时 +0.02~0.04 |
| 混凝土（干燥、粗糙） | 0.89–0.94 | 0.90–0.95 | 与含水率强相关 |
| 草地（茂密绿色） | 0.96–0.98 | 0.95–0.98 | 枯草 0.88–0.94 |
| 海水（天底方向） | 0.96–0.98 | 0.96–0.98 | 掠射角 $>60^\circ$ 时 $\to1.0$ |
| 玻璃（普通钠钙窗玻璃） | **0.85–0.92（不透明）** | 0.87–0.93 | **见下注** |
| 雪 | 0.96–0.99 | 0.96–0.99 | |
| 人体皮肤 | 0.97–0.98 | 0.97–0.98 | |

三条必须注意的：

1. **玻璃**：普通窗玻璃 >2.5 µm 已完全不透明（SiO₂ 晶格吸收），所以 MWIR 和 LWIR 都应按"不透明、$\varepsilon\approx0.9$"处理——现状 `materials.py` 里 `glass: emissivity_mwir=0.70` 隐含 30% 透射，会低估玻璃的辐射。真正的 3–5 µm 透明材料是蓝宝石/石英/Ge/ZnSe/Si，那是**镜头**材料不是场景材质。**Low-E 玻璃 LWIR $\varepsilon\approx0.1$–0.2**，与普通玻璃差 5 倍，做城市场景时值得单列一个材质。
2. **金属必须按镜面处理**。$\varepsilon$=0.08 的裸铝，92% 的信号来自 $(1-\varepsilon)L_{\rm env}$。若像现在这样用半球平均的 $L_{\rm env}=B(T_{\rm air})$，铝板会恒等于气温；真实情况是它反射**天空**（冷，MWIR 差 60 K）或**地面**（暖），随朝向剧烈变化。这是金属目标在 IR 图像里"忽冷忽热"的主因。
3. $\varepsilon$ 有角度依赖：金属在 $>60^\circ$ 掠射时 $\varepsilon$ 显著上升（可到 0.3–0.5），非金属基本不变。远景地面/水面因此常偏亮。

---

## 6. 现状诊断清单（逐条落到代码）

### A 级：物理错误，会系统性污染训练集

| # | 位置 | 问题 | 后果 |
|---|---|---|---|
| A1 | `ir.py:169` | **完全没有路径辐射项** | LWIR 绝对亮温偏冷 10–40 K，MWIR 偏冷 3–10 K；偏差随距离单调增长，模型会学到"远=冷" |
| A2 | `ir.py:173`、`weather.py:148` | 宽带 $T_{\rm sky}$ 直接代入波段普朗克 | MWIR 天空辐射高 3–8 倍，天空背景虚高 ~50 K；LWIR 高 1.7 倍 |
| A3 | `weather.py:27,163` | 常数消光 0.10/0.06 + 能见度 0.3 次幂 | 10 km 处透过率误差 2–4 倍；LWIR 与能见度的耦合方向错误 |
| A4 | `thermal.py:24`、`weather.py:148` | Swinbank 乘 $(1-0.25{\rm RH})$ | 晴空 $T_{\rm sky}$ 偏低 25–35 K（$\varepsilon$ 从 0.84 掉到 0.48）；地表系统性偏冷 5–7 K；**物理方向也反了**（越湿天空越暖） |

### B 级：破坏昼夜/方位一致性

| # | 位置 | 问题 | 后果 |
|---|---|---|---|
| B1 | `weather.py:142-145` | 系数 0.75（Meinel 原式 0.7）、$m=1/\sin$ 在低太阳角发散、$G_{\rm dir}$ 已含 $\sin(el)$ 后面又乘 $\cos\theta$（**双重计数**） | 斜面上太阳辐照偏高 2–5 倍；日出日落时天空/地面亮度突变 |
| B2 | `weather.py:145` | 云衰减 $(1-0.85C^{2.5})$ 纯经验、无物理依据 | 云的透射关系不可外推 |
| B3 | `thermal.py:96` | 无**漫射太阳**（只有直射 × shadow） | 背阴面在白天是纯黑的，和真实（散射天空照亮）差别很大 |
| B4 | `ir.py:169` | $(1-\varepsilon)L_{\rm env}$ 用半球平均且不区分方向 | 金属/玻璃/水面的镜面反射完全丢失 |
| B5 | `ir.py:179` | NETD 用全局平均 $\partial L/\partial T$ | 高温区噪声被高估、冷区被低估，噪声与信号的相关性错误 |

### C 级：真实性细节

| # | 位置 | 问题 |
|---|---|---|
| C1 | `thermal.py:88` | $h=5.7+3.8v$ 是 Jürges 强制对流式（$v>1$ m/s），缺自然对流下限；静风时应取 $h=\max(5.7+3.8v,\ 1.31\Delta T^{1/3}+\ldots)$ |
| C2 | `thermal.py` 全文 | 逐面稳态、无热惯性、无面间导热；$q_{\rm int}$ 是常数。对静止场景可接受，对"车辆刚停下/驶入阴影"这类时序任务是错的 |
| C3 | `thermal.py:93` | 用 $h$ 加 8+15v 来模拟蒸发，是量纲不严的经验补丁；蒸发潜热应作为 $q_{\rm lat}=\lambda_E\dot m$ 单列 |
| C4 | `ir.py:191` | 16 bit 编码线性于**温度**而非辐射/电子数 |
| C5 | 全局 | 无光谱加权（用整段平均 $\varepsilon$、$\tau$），未乘探测器响应曲线；宽波段近似下 MWIR 的 4.3 µm 死区与 3.7 µm 窗口被同等对待 |
| C6 | 全局 | 无大气湍流 MTF、无光学系统的衍射/像差 MTF、无采样相位效应 |

---

## 7. 参考实现（可直接替换 `weather.py derive()` 与 `ir.py` 的核心）

```python
import numpy as np

SIGMA = 5.670374419e-8
C1, C2 = 1.191042972e-16, 1.4387769e-2      # Planck 常数组合（SI）

def planck(lam_um, T):
    """光谱辐射亮度 W m-2 sr-1 um-1"""
    lam = np.asarray(lam_um, float) * 1e-6
    return C1 / lam**5 / np.expm1(C2 / (lam * T)) * 1e-6

def planck_band(T, l1, l2, n=64):
    lam = np.linspace(l1, l2, n)
    return np.trapezoid(planck(lam, T), lam) if hasattr(np,'trapezoid') else np.trapz(planck(lam, T), lam)

# ---------- 1. 大气 ----------
def kruse_q(V_km):
    if V_km > 50:  return 1.6
    if V_km > 6:   return 1.3
    if V_km > 1:   return 0.16*V_km + 0.34
    if V_km > 0.5: return V_km - 0.5
    return 0.0

def beta_aer(lam_um, V_km):
    return (3.912 / max(V_km, 0.05)) * (lam_um / 0.55) ** (-kruse_q(V_km))

MOL = {   # (k0 [1/km], k1 [1/km per cm])
    "mwir": (0.28, 0.06),
    "lwir": (0.02, 0.13),
}

def pwv_cm(T_air_C, rh, R_km):
    es = 6.112 * np.exp(17.62*T_air_C/(243.12+T_air_C))     # hPa
    e  = np.clip(rh,0,1) * es
    rho_v = 216.7 * e / (T_air_C + 273.15)                  # g/m3
    return 0.1 * rho_v * R_km                               # cm

def extinction(band, lam_um, T_air_C, rh, R_km, V_km):
    w = pwv_cm(T_air_C, rh, R_km)
    k0, k1 = MOL[band]
    return beta_aer(lam_um, V_km) + k0 + k1 * w

def transmittance(band, lam_um, R_km, T_air_C, rh, V_km):
    return np.exp(-extinction(band, lam_um, T_air_C, rh, R_km, V_km) * R_km)

# ---------- 2. 天空 ----------
def eps_sky_clark_allen(T_air_C, dewpt_C, cloud_frac):
    Td = dewpt_C + 273.15
    e_clr = 0.787 + 0.764*np.log(Td/273.0)
    e_clr = np.clip(e_clr, 0.5, 1.0)
    N = np.clip(cloud_frac, 0, 1) * 10.0
    return np.clip(e_clr*(1 + 0.0224*N - 0.0035*N**2 + 0.00028*N**3), 0, 1)

KAPPA = {"mwir": 0.30, "lwir": 0.62}     # 波段折扣（见 2.2）
def sky_radiance(band, l1, l2, T_air_K, eps_sky):
    return KAPPA[band]*eps_sky*planck_band(T_air_K, l1, l2)

# ---------- 3. 辐射 ----------
def at_sensor(L_tgt, tau, L_path):
    return tau*L_tgt + L_path

def path_radiance_iso(tau, T_air_K, l1, l2):
    return (1.0 - tau) * planck_band(T_air_K, l1, l2)

def solar_scatter(E0, mu0, mu, tau_sca, cos_theta, w0=0.8, g=0.6):
    P = (1-g*g) / (1 + g*g - 2*g*cos_theta)**1.5
    x = tau_sca*(1.0/mu0 + 1.0/mu)
    return w0*E0*mu0*P/(4*np.pi) * (-np.expm1(-x))/(1.0/mu0 + 1.0/mu)
```

**自检**（放进 `selftest.py`）：

| 检查项 | 期望 |
|---|---|
| $T$=300 K，8–12 µm，$B_{\rm band}$ | 38.5 W m⁻² sr⁻¹（±1%） |
| $\partial L/\partial T$ 同条件 | 0.63 W m⁻² sr⁻¹ K⁻¹ |
| $\tau$(MWIR, 3 km, 20 °C, 60% RH, V=20 km) | 0.34–0.40 |
| $\tau$(LWIR, 3 km, 同条件) | 0.55–0.62 |
| 晴空 $T_{\rm sky}$（20 °C, 露点 10 °C, 全波段） | 281–284 K |
| $\kappa_{\rm MWIR}$ / $\kappa_{\rm LWIR}$ | 0.20–0.30 / 0.45–0.60 |
| $L_{\rm path}$/总信号（LWIR, 3 km, τ≈0.62） | 0.35–0.45（MWIR 同条件约 0.6） |
| FPN 图案的相关长度 | 3–10 px |

---

## 8. 出处

**大气**
- Kruse, McGlauchlin & McQuistan, *Elements of Infrared Technology*, Wiley, 1962（Kruse 消光/能见度式）。
- Kim, McArthur & Korevaar, "Comparison of laser beam propagation at 785 nm and 1550 nm in fog and haze for optical wireless communications", *Proc. SPIE* 4214, 2001（$q$ 分段修正）。国产 FSO 综述复述见 <https://doi.org/10.3390/telecom6030057>。
- Kneizys et al., *LOWTRAN 7 User's Guide*, AFGL-TR-88-0177, 1988；Berk et al., *MODTRAN4 User's Manual*, 1999。
- 免费 MODTRAN Web 版：<http://modtran.spectral.com/modtran_home>
- 实测双指数拟合与 3–5 µm 透过率数据：中国专利 [CN108375554A](https://patents.google.com/patent/CN108375554A/zh)（西安应用光学研究所，2018）。
- 热红外通道透过率经验模式（MODTRAN 查表 + 拟合）：《热红外遥感中大气透过率的研究（一）》红外与激光工程，<http://hwyjggc.xml-journal.net/article/id/1859>

**天空/长波辐射**
- Swinbank, *QJRMS* 89:339, 1963。
- Brunt, *QJRMS* 58:389, 1932。
- Idso & Jackson, "Thermal radiation from the atmosphere", *JGR* 74:5397, 1969，<https://doi.org/10.1029/JC074i023p05397>
- Brutsaert, *Water Resour. Res.* 11:543, 1975。
- Berdahl & Martin, "Emissivity of clear skies", *Solar Energy* 32:663, 1984，<https://doi.org/10.1016/0038-092X(84)90144-0>
- Clark & Allen, "The estimation of atmospheric radiation for clear and cloudy skies", *Proc. 2nd Nat. Passive Solar Conf.*, 1978；EnergyPlus 实现见 Walton, NBSIR 83-2655, 1983 与 EnergyPlus Engineering Reference。
- Martin & Berdahl, "Characteristics of infrared sky radiation in the United States", *Solar Energy* 33:321, 1984，<https://doi.org/10.1016/0038-092X(84)90162-2>

**太阳**
- Meinel & Meinel, *Applied Solar Energy*, Addison-Wesley, 1976（$0.7^{m^{0.678}}$）。
- Kasten & Young, "Revised optical air mass tables and approximation formula", *Appl. Opt.* 28:4735, 1989，<https://doi.org/10.1364/AO.28.004735>
- 散射相函数与单次散射：Liou, *An Introduction to Atmospheric Radiation*, 2nd ed., 2002, Ch. 6。

**探测/噪声**
- Hudson, *Infrared System Engineering*, Wiley, 1969（NETD–D* 关系）。
- Dereniak & Boreman, *Infrared Detectors and Systems*, Wiley, 1996。
- NVTherm 手册（3D 噪声、MRTD、大气透过率 LUT）：<https://zieg.com/links/sensors/NVThermManual5.pdf>
- Driggers et al., "Review of infrared scene projector science and technology", *Opt. Eng.*（FPN/NUC 残差量级）。

**材质**
- Wolfe & Zissis, *The Infrared Handbook*, 3rd ed.；Accetta & Shumaker, *The Infrared and Electro-Optical Systems Handbook*, Vol. 3。
- 实测表：<https://www.thermoworks.com/blogs/learning-center/emissivity-table>、<https://www.transmetra.ch/images/transmetra_pdf/publikationen_literatur/pyrometrie-thermografie/emissivity_table.pdf>、<https://ennologic.com/wp-content/uploads/2018/07/Ultimate-Emissivity-Table.pdf>

---

## 9. 一句话总结

**最该改的三件事**：加路径辐射 $(1-\tau)B(T_{\rm air})$、天空改成波段折扣 $\kappa_{\rm band}B_{\rm band}(T_{\rm air})$、消光系数换成 Kruse + 水汽（MWIR 0.35、LWIR 0.16 km⁻¹ 量级）。
这三处改完，仿真器的**绝对辐射定标**才成立；其余（噪声、FPN、镜面反射、漫射太阳）是让训练集"像真实传感器"的第二梯队。
