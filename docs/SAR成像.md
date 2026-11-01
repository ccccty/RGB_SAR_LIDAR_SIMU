# SAR 成像（条带式 Stripmap + 距离-多普勒）

`simkit/sar.py` 是这套仿真里唯一的 SAR 前向模型。它把 `scene.build_world()` 的三角面世界变成一张定标的 σ0 灰度图，并给出每像素属于哪个目标（`obj_map`），供检测数据集直接使用。

对外唯一入口：

```python
from simkit import scene, sar
w = scene.build_world(cfg, t=0.0, jitter=jit)
out = sar.simulate(w, cfg)          # cfg["sensors"]["sar"]
```

---

## 1. 为什么不是「逐脉冲射线」

真实 SAR 收的是 `n_pulses × n_samples` 个复采样，然后做二维匹配滤波。直接照这个来，一帧 512×1024 就要 5×10⁵ 条射线的**相位历史**，每秒几十亿次三角求交，纯 numpy 一分钟出一帧，数据集没法用。

这里用的是严格等价的**距离-多普勒域**写法。正侧视条带几何下，地面散射点随方位偏移 `u` 的双程相位是

```
φ(u) = -4π·sqrt(R0² + u²) / λ
```

对固定地距 `R0` 它**平移不变**，所以二维匹配滤波 = 场景反射率与压缩后点扩散函数 `k ⊛ k*` 的卷积，而 `k ⊛ k*` 是宽度由相位历史曲率决定的 sinc。于是：

* 每像素一条射线 → 得到该分辨率单元的复反射率（含 PO RCS、材质 ρ、粗糙度）；
* 一次可分离卷积 → 完成距离压缩 + 方位压缩；
* 相位历史的**曲率**仍然真正被用到：方位分辨率由它算出（见 §3），不是拍脑袋常数。

结果：3–4 s/帧、665–690 MB 峰值，且分辨率、SNR、几何都对得上物理量（§7 有实测）。

## 2. 成像链路

```
平台几何（heading / squint / altitude / speed）
   ↓  斜距均匀网格 + 方位网格
每条射线：BVH 求交 → 命中三角面 → 面积/入射角/材质 ρ/粗糙度
   ↓  facet_sigma()（与 simkit/radar.py 同一套 PO 公式）
γ0(x,y)  [无量纲，RCS / 地面 m²]
   ↓  子像素目标积分（每物体 4× 细格，按命中点入箱）
γ0 + 物体 id
   ↓  方位 sinc × 距离 sinc（-3 dB 宽度 = ρ_a, ρ_r）
聚焦幅度 A(x,y)
   ↓  复斑点场（同一 PSF 卷积后调制包络）+ 热噪声
功率 P = A²·|speckle|² + NESZ·|n|²
   ↓  10log10
image_db (dB, σ0 定标) → 显示拉伸 → image_u8
```

### 2.1 几何

* 平台沿 `heading_deg` 直线飞行（**不是**跟着目标拐弯——否则条带就是聚束了），速度 `speed_mps`，高度 `altitude_m`。
* 视向：`heading` 侧向 ±90°，再按 `squint_deg` 往机头方向偏（90° = 正侧视）。
* 姿态锚点：`sensors.sar.anchor` 指定帧 0 时刻测绘带中心瞄准的地面点；不给就用 `cfg["objects"]` 里**配置位置**（不是当前位姿）的质心。这一点很关键：如果按每帧目标实际位置重锚，平台会追着目标飞，帧间就没有条带 SAR 应有的地物滑动。
* **每帧位姿**：如果 `sensors.sar` 里带 `pos`（3 元组），它**就是孔径中心**，不再由 `anchor`+`altitude_m` 反推；同时有效高度取 `pos[2] - ground_z`，`pos` 已含该帧的位移，不会再叠 `vel·t`（否则与 `plan.py` 走出的位姿重复叠加）。`look_at` 若存在，视向直接由 `pos → look_at` 决定，左右侧别与 `squint` 由它**反解**（`squint = 90° + atan2(沿航迹分量, 侧向分量)`）；`heading_deg`/`speed_mps` 仍然只决定航迹与速度。这样 `dataset.py` 每帧改写 `pos/look_at` 后，测绘带跟着平台走，跨模态配准才对得上。没有 `pos` 时退回上面的几何构造（独立工具脚本走这条）。<br>`platform.pose_source` 会写 `"config"` 还是 `"derived"`，`platform.side` 是实际生效的侧别。
* 列方向在**斜距**上均匀（雷达实际测的就是斜距），`range_axis_m` 汇报的是对应的地距 `sqrt(R²-h²)`（近→远单调），另给 `range_slant_axis_m`。这样距离 PSF 均匀，且高程目标落在它自己的斜距上 = 标准**叠掩向近距**；若按地距均匀建格，目标反而会被推到远距。
* 高程目标会被射线在更高处截住，其后的地面落在阴影里（`γ0=0`），图上就是「亮点 + 黑带」。这是 SAR 本来的样子，不是 bug。

### 2.2 分辨率

```
ρ_r = c / (2B)                       斜距向 -3 dB 宽（乘 resolution_scale）
ρ_a = 0.886 · λ·R0 / (2·L_sa)        方位向
L_sa = min(v·n_pulses/prf, λ·R0/L_ant)
```

`L_sa` 的第二个限制是实孔径波束足迹：天线再长也不会让相干孔径无限长，所以一旦波束受限，`ρ_a → 0.886·L_ant/2`。

方位分辨率的推导走的是相位历史：`|d²φ/du²| = 4π/(λR0)` → 多普勒调频率 `K_a = k·v²/2π` → 孔径内多普勒带宽 `B_d = K_a·T_int` → `ρ_a = 0.886·v/B_d`。

#### D/2 定律：天线越长，方位分辨率越差（反直觉，务必读完再改）

这是条带 SAR 最容易写反的一条。直觉是「天线越大波束越窄、应该看得越清」，但对**条带** SAR 恰好相反：

```
ρ_a = 0.886 · L_ant / 2          （波束受限区，即真实系统的常态）
```

原因不在天线本身，在**相干积累时间**。方位压缩的带宽来自散射点在合成孔径内划过的多普勒历程，而它能被相干积累的时长受实波束驻留时间限制：波束越窄（天线越长），目标被照到的时间越短，可用的合成孔径越短，压缩后的主瓣越宽。天线加长一倍，分辨率就粗一倍——这是代价，不是 bug。

实测（X 波段 9.6 GHz、R0 = 3.2 km、v = 120 m/s、PRF 1200、512 脉冲，`selftest()` 第 2 项原文）：

| `antenna_len_m` | `ρ_a` |
|---|---|
| 1.0 m | 0.86 m |
| 4.0 m | 1.77 m |

**想提高方位分辨率，正确做法是加长相干积累时间，不是加长天线**：

* `n_pulses` ↑ 或 `prf_hz` ↑（同一 CPI 内更多脉冲）；或
* 平台速度 ↓（同样脉冲数覆盖更长的空间孔径，但注意这是同一件事的另一种写法：`L_sa = v·n_pulses/prf`）。

实测（`antenna_len_m = 0.5 m`，其余同上）：`n_pulses` 256 → 1.73 m，1024 → 0.43 m（合成孔径 26 m → 102 m）。

`selftest()` 用两条分开命名的断言把两件事钉住，避免读者误以为互相矛盾：

```
PASS  azimuth coarsens with antenna length (D/2 law)
PASS  azimuth sharpens with integration time
```

> 历史注记：契约初稿写的是「天线变长 → 方位分辨率数值变小」，方向写反了（想表达的其实是「合成孔径变长 → 分辨率变好」）。已与主控确认按物理实现 (a)，契约原文由主控修正，实现不为迎合文本编造核宽度。

### 2.3 反射率与定标

* 每个分辨率单元：`facet_sigma(面积, cos入射, ρ, rough, λ, footprint=单元投影面积)`，与 `radar.py` 完全同一套 PO（相干项 ∝ 4πa²cos²/λ²·ρ²·exp(-(4π·rough·cos/λ)²) + 漫反射项）。地面的粗糙度让相干项塌掉、只剩漫反射，所以地物是「散斑」而不是镜面。
* 归一化：`γ0 = σ / 地面单元面积`（`cell_area` 逐列算，因为斜距均匀网格在地距上非均匀）。
* 子像素目标：单条射线可能整个跳过一块比像素小的金属板，且命中与否取决于亚米级网格相位，帧间会闪。因此每个物体在其包围盒内**再采样 4× 细格**，按命中点自己的 `(u, R)` 入箱，`γ0 = Σσ / 单元面积`（能量守恒）。这样点目标亮度与网格相位无关。

### 2.4 雷达方程与 NESZ

```
G    = 4π·A·η / λ²                     A = antenna_len × antenna_h
Pr   = Pt·G²·λ²·σ / ((4π)³·R⁴·L)
SNR  = N_pulses · (T_p·Pr) / (k·T0·F)  T_p = duty/prf（脉压增益 T_p·B）
NESZ = R⁴·k·T0·F / (N_pulses·T_p·K_sys·A_cell·att)
```

图像以 σ0 为单位成像，所以噪声底在图上就等于 NESZ（`stats.noise_db = 10log10(NESZ)`），目标峰值就是该目标的 γ0。`stats.snr_db = target_peak_db - noise_db`，等价于「目标 σ0 相对 NESZ 的余量」。发射功率 +3 dB → NESZ 降 3 dB → `snr_db` +3 dB（selftest 实测 +3.01）。

大气：读 `cfg["_weather"]["radar_att_db_km"]`（`weather.derive()` 给的 ITU-R P.676/P.838），双程衰减。

### 2.5 斑点

斑点来自一个分辨率单元内大量亚分辨散射体的干涉，因此它的相关长度**就是** PSF 宽度。做法是：生成单位功率复高斯场 → 用同一对 PSF 核卷积 → 归一化到单位均值功率 → 调制包络。

如果按像素独立抽指数分布，得到的是白噪声椒盐，视觉上像传感器噪声而不像 SAR，且空间统计（自相关长度、等效视数）全错——检测器学不到东西。

`looks` 做多视平均（`looks=1` 单视，方差最大；`looks=4` 视觉更干净）。

### 2.6 显示

`image_u8`：以 10% 分位为锚 + 固定 `display_dyn_range_db`（默认 26 dB）+ gamma 0.8。与 `dataset.py` 的 `scope()` 同一思路，只把锚点从中位数改成低分位——聚焦后的 SAR 图里中位数是**地物**不是噪声，按中位数锚会把一半地物压到黑。`image_db` 保留标定值不动。

### 2.7 目标框：`obj_map` + `target_boxes` 双保险

`obj_map` 是逐像素的：每个物体的反射率单独过同一 PSF，逐像素取能量最大者，背景 -1。它在目标足够大时非常准（bbox 中心与几何解算差 ≤1 px）。

问题是**小目标**：4 m 的车辆在 0.8 m 分辨率下占 25 个分辨单元没问题，但 0.4 m 的板子只落在 0–2 个像素里，`labels.boxes_from_mask(min_px=3)` 会直接把它丢掉。

因此 `simulate()` 额外返回 `target_boxes = {物体名: [x0, y0, x1, y1]}`（**像素坐标，x = 列 = 距离向，y = 行 = 方位向；x1/y1 为开区间上界**，与 `labels.boxes_from_mask()` 同一套半开约定，可直接塞进 box dict 不用 +1）：

* 由物体 8 个角点按 `(u = Δ·ĥd, R = |Δ|)` 直接投影，不经过成像，所以**掩膜为空时它也在**；
* 与 `obj_map` 的掩膜框取并集，两者谁都丢不了目标；
* 只对 `label: true` 且 `visible: true` 的物体生成——scenery 照样成像、照样有 `obj_map` id，但不进标注。

`tools/sar_check.py` 用的是同一套兜底：掩膜像素 ≥4 用掩膜框（`source="mask"`），否则用投影框（`source="geometry"`），并在输出里打印掩膜实际像素数。

## 3. 参数

`simkit/sar.py` 的 `DEFAULTS`，场景里写 `"sensors": {"sar": {...}}`，同名键覆盖。

| 契约键 | 默认 | 说明 |
|---|---|---|
| band | X | X/C/Ku/L/S 仅作 fc 预设提示 |
| fc_ghz | 9.6 | 载频 |
| bandwidth_mhz | 150 | 决定 ρ_r = c/2B |
| prf_hz | 1200 | 脉冲重复频率 |
| n_pulses | 512 | 相干积累脉冲数 → L_sa |
| n_samples | 1024 | 距离向像素数（列数） |
| altitude_m | 2000 | 平台高度 |
| speed_mps | 120 | 地速 |
| heading_deg / squint_deg | 90 / 90 | 航向 / 斜视（90=正侧视） |
| swath_center_m / swath_width_m | 2500 / 560 | 测绘带（**地距**，米） |
| polarization | VV | 仅标注，不改物理（见 §9） |
| tx_power_w | 40 | 峰值功率 |
| noise_figure_db / loss_db | 4 / 3 | 接收机 |
| antenna_len_m / antenna_h_m | 1.2 / 0.35 | 方位/俯仰孔径 |
| resolution_scale | 1.0 | 分辨率乘子（同时作用于 ρ_r 与 ρ_a） |
| **扩展键** | | |
| n_cross | 512 | 方位向像素数（行数） |
| az_extent_m | 0 | 方位覆盖（0=按 n_cross·ρ_a/oversample 自动） |
| oversample | 2.0 | 像素间距 = ρ/oversample（2 = 常规过采样，便于显示与测宽） |
| looks | 1 | 多视数 |
| duty | 0.08 | 占空比 → T_p = duty/prf |
| antenna_eff | 0.6 | 孔径效率 |
| specular_scale_db | -20 | 平facet相干项标定（同 radar.py） |
| diffuse_gamma | 0.06 | 漫反射 γ0 系数 |
| ground_clutter_db | 0 | 背景地物整体电平（雷达散射标定） |
| side | right | left/right（有 `look_at` 时由它反解，此键被忽略） |
| anchor | null | 测绘带瞄准点（有 `pos` 时被忽略） |
| pos | null | **孔径中心**（3 元组，米）。`dataset.py` 可选地每帧写成本帧位姿；给了就直接用、不再叠加 `vel·t` |
| look_at | null | 视向参考点；与 `pos` 同时存在时决定左右侧与 squint |
| display_dyn_range_db / display_anchor_pct / display_gamma | 26 / 10 / 0.8 | 显示 |
| speckle / thermal_noise | true / true | 关掉得到确定性冲激响应（标定/测 PSF 用） |
| seed | 0 | 未传 rng 时的种子 |

## 4. `simulate()` 返回

严格按契约（多给几个便于自证/展示的键）：

| key | 说明 |
|---|---|
| `image_u8` | (n_cross, n_range) uint8，显示用 |
| `image_db` | float32，σ0 定标 dB（噪声底 = NESZ） |
| `obj_map` | int32，每像素目标 id，背景 -1 |
| `range_axis_m` | (W,) 地距（近→远） |
| `range_slant_axis_m` | (W,) 斜距，**列在斜距上均匀** |
| `cross_axis_m` | (H,) 方位坐标（沿航迹，米） |
| `resolution` | `range_m`(地距), `azimuth_m`, `range_slant_m`, `l_sa_m` |
| `swath` | `near_m/far_m/width_m/center_m/incidence_deg/range_sampling_m/sampling_ratio/near_slant_m/far_slant_m/ground_sampling_m` |
| `platform` | `pos/vel/altitude_m/heading_deg/squint_deg/t_s/side/pose_source` |
| `stats` | `peak_db, mean_db, noise_db(NESZ), snr_db, dynamic_range_db, n_scatterers, target_peak_db, target_contrast_db, target_mean_db, nesz_db, gain_db, cell_area_m2, looks` |
| `target_boxes` | `{物体名: [x0,y0,x1,y1]}` 像素坐标（x 列/距离，y 行/方位，**x1/y1 开区间**），只含 `label:true` 物体；空场景为 `{}` |
| `scatterers` | 最多 2000 个 `{x,y,z,sigma,obj}`，按 σ 降序 |

`target_contrast_db` 用**峰值减背景中位数**（标准 SCR）。用盒内均值会把「框里恰好有多少雷达阴影」算进指标，不稳。

## 5. 用法

```bash
# 单帧复核：出图 + 标签 + 打印分辨率/测绘带/stats/bbox/耗时/内存
PYTHONPATH=. ./env/Scripts/python.exe tools/sar_check.py -c scenes/sar_air_uav.json -o out/sar_check

# 物理自检（9 项断言，返回失败项数）
PYTHONPATH=. ./env/Scripts/python.exe -c "from simkit import sar; raise SystemExit(sar.selftest())"

# 场景本身是否被 planner 接受
PYTHONPATH=. ./env/Scripts/python.exe -m simkit plan -c scenes/sar_ground_vehicle.json
```

`tools/sar_check.py` 输出：

```
<out>/images/sar/000000.png       8-bit 显示图
<out>/labels/sar/000000.json      分辨率/测绘带/平台/stats/box/target_boxes/真值位置
<out>/sar_debug/000000_db16.png   16-bit 标定强度（不裁到 8-bit）
```

## 6. 场景

| 场景 | 平台 | 目标 | 分辨率 | 方位覆盖 | SCR |
|---|---|---|---|---|---|
| `scenes/sar_air_uav.json` | 无人机载 SAR，H=1200 m，v=25 m/s，X 波段 250 MHz | 3.6 m 四旋翼，海面上空 30 m | 0.75 m × 0.81 m | 512 行 × 0.40 m = 202 m | +25.7 dB |
| `scenes/sar_ground_vehicle.json` | H=3000 m，v=40 m/s，X 波段 300 MHz | 4.4 m 车辆（另有两件 scenery） | 0.78 m × 0.79 m | 640 行 × 0.63 m = 400 m | +28.7 dB |

**方位覆盖要按 `frames` 算**：测绘带跟着平台走，地面目标就会沿方位向滚出图。`v·dt·(帧数-1)` 必须明显小于方位覆盖的一半，否则最后几帧目标会贴边或丢失。ground 场景原本 512 行 × 0.40 m = 202 m，40 m/s × 0.5 s × 5 帧 = 100 m 正好把它推出画面；加到 640 行、400 m 后 6 帧都留在图内（rows 315→156）。

两场景都只放**一个** `label:true` 的对象：`plan.py` 会把所有 `label:true` 物体一起拖到同一条目标轨迹上（`ClipPlan.state()` 的既有行为），多目标必须走 scenery（`label:false`）——它们照样成像、照样有 `obj_map` id，只是不进标签。

## 7. 实测（X 波段，Windows，`./env/Scripts/python.exe`）

`selftest()` 9/9 PASS：

```
PASS  range resolution = c/2B                measured -3dB 1.014 m (slant) vs theory 0.999 m, err +1.5%
PASS  azimuth coarsens with antenna length (D/2 law) L=1.0 m -> 0.86 m, L=4.0 m -> 1.77 m (longer antenna = coarser azimuth)
PASS  azimuth sharpens with integration time n_pulses 256 -> 1.73 m, 1024 -> 0.43 m (aperture 26 -> 102 m)
PASS  point target peak at geometric position expected (row 178, col 126) for u=60.0 m g=2500.0 m, peak (row 178, col 127), err 0/1 px
PASS  +3 dB tx power -> +3 dB SNR            SNR 56.61 -> 59.62 dB (delta +3.01 dB), NESZ -67.6 -> -70.6 dB
PASS  in-footprint ground echo > outside     covered -21.8 dB vs bare -75.0 dB (53.2 dB)
PASS  target bbox on target, contrast > 0    bbox rows 178-178 cols 126-128 (centre 178.0,127.0) vs true (178,126), offset 0.0/1.0 px, contrast +13.8 dB
PASS  target_boxes covers sub-resolution target mask 0 px, box [31, 24, 32, 25], true pixel (row 25, col 31)
PASS  sensors.sar.pos is the aperture centre pos honoured (err 0 m), alt 2000.0 m, side right, peak (row 38, col 126) vs geometry (38, 126)
9/9 passed
```

第 8 项是 `target_boxes` 的兜底能力本身：4 m × 6.4 m 的像素对 0.4 m 的板子，`obj_map` **一个像素都没有**（`mask 0 px`），投影框照样存在且框住了真实解算位置。

场景复核（`tools/sar_check.py`）：

| | sar_air_uav | sar_ground_vehicle |
|---|---|---|
| 图像 | 512×768 | 640×768 |
| ρ_r（斜距 / 地距） | 0.60 / 0.75 m | 0.50 / 0.78 m |
| ρ_a | 0.81 m（孔径 34.1 m） | 0.79 m（孔径 68.3 m） |
| 测绘带 | 1400–1800 m，入射 53.1° | 2300–2700 m，入射 39.8° |
| 距离采样 | 0.42 m/样（1.4×） | 0.33 m/样（1.5×） |
| 方位采样 | 0.40 m/行（2.0×） | 0.63 m/行（1.3×） |
| peak / mean / NESZ | +3.7 / −22.4 / −60.7 dB | +6.6 / −23.0 / −59.0 dB |
| SNR | 64.3 dB | 65.6 dB |
| 目标峰值 / SCR | +3.7 / +25.7 dB | +6.6 / +28.7 dB |
| **目标占多少像素** | 掩膜 62 px，框 8×12（x327–335, y250–262，半开） | 掩膜 35 px，框 6×8（x370–376, y316–324） |
| bbox 与几何解算偏差 | 0.0 px / −0.42 m | 1.0 px / −0.33 m |
| 耗时（单次，空载） | 3.78 s wall / 3.73 s CPU | 2.68 s wall / 2.69 s CPU |
| 峰值工作集 | 665 MB | 689 MB |

NESZ 到 −60 dB 是因为「1–3 km 近距 + 2000 脉冲相干积累」，短距压倒一切；这不是星载量级（星载 −25 dB 左右），量级是对的。

耗时：`ray_cast` 一项占 ~85%（`bvh.py` 的批量遍历，不是本模块），所以耗时≈(n_cross·n_samples) 线性。空载 2.9–3.6 s；机器被 Blender/无头浏览器占满时同一代码会涨到 6.6–7.1 s CPU / 8–9 s wall——这是 CPU 争用，不是算法变慢（分块调用反而更慢 50%，已回退）。

## 8. 主控集成点（我不改这些文件）

1. `simkit/dataset.py`：`sensors` 白名单里加 `"sar"`；分支写成

   ```python
   if "sar" in sensors:
       from . import sar
       # 每帧位姿：把该帧的真实位姿写进传感器块，sar 会直接拿它当孔径中心
       fr_cfg["sensors"]["sar"]["pos"] = [float(v) for v in pose_pos]
       fr_cfg["sensors"]["sar"]["look_at"] = [float(v) for v in pose_look]
       out = sar.simulate(w, fr_cfg)
       png.write_gray8(os.path.join(out_dir, "images/sar", name), out["image_u8"])
       boxes = labels.boxes_from_mask(out["obj_map"], w, min_px=3)
       # 小目标兜底：掩膜不足 min_px 的物体用 out["target_boxes"][name]
       # labels/sar/<f>.json = dict(frame, sensor="sar", image, width, height,
       #   axes, resolution, swath, platform, stats, boxes, target_boxes, objects)
   ```
   注意 `scene.build_world(...)` 那一段已经覆盖 `{"ir","radar","lidar","sar"}`，无需再动。

   **`pos` / `look_at` 的语义**（重要）：`pos` 是**孔径中心本身**，已含该帧位移，`simulate()` 不会再叠加 `vel·t`；有效高度取 `pos[2] - ground_z`（所以地面高度非 0 时也自洽）；`look_at` 决定视向，左右侧与 squint 由它反解。不写这两个键则退回 `anchor`/`heading_deg` 几何，行为与之前完全一致（`platform.pose_source` 会写明走的是哪条）。

2. `sv_dict`/manifest：按契约 §3 的 `frame.sar = {near_m, far_m, ..., platform{...}}` 直接取 `out["swath"] / out["resolution"] / out["stats"] / out["platform"]`，字段名一一对应（`res_range_m ← resolution.range_m`、`res_az_m ← resolution.azimuth_m`）。`boxes` 里的 `x0/y0/x1/y1` 已经是像素坐标。

3. `simkit/cli.py`：`-s` 的 choices 加 `sar`。

4. `simkit/selftest.py`：挂 `from .sar import selftest as sar_selftest`，把它加进失败计数；`sar.selftest()` 返回失败项数（0 = 全过），自己会打印 PASS/FAIL 行。

5. 网页端注意：SAR 图的**列在斜距上均匀**，`range_axis_m` 是地距刻度（非线性），画刻度尺请用 `range_slant_axis_m` 或按 `m = sqrt(R²-h²)` 换算。

## 9. 已知近似与限制（老实写）

1. **单次反射**：无二次散射、无腔体/角反射器谐振。对车辆、建筑的 RCS 会偏低（真实车辆靠底盘-地面多次反射往往比单次 PO 亮 5–10 dB）。
2. **平facet PO + 全局 `specular_scale_db=-20 dB`**：沿用 `radar.py` 的标定，实际是经验系数，不是严格 PO 积分；掠射角下仍偏乐观。
3. **无距离徙动 / 无方位模糊**：假定 PRF 满足 Nyquist、孔径内徙动小于一个距离门。宽波束或大斜视会失效（场景里 `squint_deg` 偏离 90° 太多时方位核宽度会不准）。
4. **确定性相位不进卷积**：完全聚焦的像里，匹配滤波会把二次相位消掉——把 `exp(-j4πR/λ)` 乘在反射率上再做卷积会重新注入相位斜坡，在地面上画出几十米周期的假栅格（第一版踩过，图上是很规整的方格）。相位历史的曲率仍然决定 `ρ_a`，只是不参与成像卷积；图像里保留的是斑点场的随机相位。
5. **阴影是几何遮挡，无绕射/多径回填**，所以场景里 `ground.enabled=false` 或物体后方是纯噪声底。
6. **子像素目标靠 4× 细格积分**，比像素还小一半以上的目标会退化成「一块亮斑」，`obj_map` 的框会比真实尺寸大约一个分辨率单元；bbox 中心偏差 ≤1 px，但尺寸不可当量测值。**掩膜为空时用 `target_boxes` 的投影框兜底**（selftest 第 8 项就是 0 px 掩膜的情形）。
7. **极化 `polarization` 只记录不建模**：没有 HH/VV 比值、没有极化熵，`rf_rho` 是标量。
8. **地形起伏只影响几何**：地面回波按平面网格射线求交，山体不会有正确的斜距压缩/叠掩（DEM 需要逐点投影，当前没做）。
9. **斑点与噪声在图像域加**：噪声本应逐脉冲进入再做脉压（会带有距离向相关性），这里按白噪声近似；`looks` 的多视也是图像域平均，与真实多视处理（子孔径）统计一致但纹理细节不同。
10. **列在斜距均匀**：与真实 SAR 产品一致，但与契约文字「列=地距」的字面读数有差异；`range_axis_m` 仍是地距单调序列，只是步长非均匀（近密远疏）。若下游非要地距等间距，应重采样，不要改网格。
11. **方位分辨率方向**：见 §2.2 的 D/2 定律与 selftest 第 2、3 项。契约原文方向写反，已由主控修正，实现按物理走。
12. **有 `pos` 时不做多普勒中心校正**：`look_at` 只用来定视向/侧别，`squint` 由它反解后仅影响 `ρ_a` 的孔径上限；没有做斜视引起的多普勒中心频率偏移与走动校正，大斜视（|squint−90°| > ~3°）时成像质量会下降。
13. **耗时依赖机器负载**：`ray_cast` 占 ~85% 的时间（`bvh.py` 的遍历，不是本模块），空载 2.9–3.6 s，机器被 Blender/无头浏览器占满时同一份代码会到 6.6–7.1 s CPU / 8–9 s wall。要更快只能降 `n_cross`×`n_samples`（射线数线性决定耗时）；试过把一次求交切成 64k 块的"优化"，实测反而慢 50%（相干射线共享节点路径，大批量摊薄了遍历的固定开销），已回退。

