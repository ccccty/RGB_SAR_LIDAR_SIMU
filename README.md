# RGB / SAR / LiDAR 多模态传感器仿真器

**输入一个 3D 模型，离线生成一整套带标注的多模态传感器数据集。**

纯 Python（numpy + Blender 无头渲染），不依赖 GPU、不联网、不需要 MATLAB，
Windows 上解压即用。面向目标检测 / 多模态融合算法的训练与验证。

> 在线演示：<https://ccccty.github.io/RGB_SAR_LIDAR_SIMU/>

---

## 1. 它做什么

给一个 OBJ 模型、一个相机位置和一组天气，输出：

| 模态 | 成像方式 | 输出 |
|---|---|---|
| **RGB 可见光** | Blender 4.5 无头 Cycles 路径追踪 | 640×480 PNG |
| **红外 MWIR** 3–5 µm | 逐面能量平衡 → Planck 波段积分 → NETD 噪声 → NUC | 16-bit PNG + 8-bit 预览 |
| **红外 LWIR** 8–12 µm | 同上，独立发射率与大气窗口 | 16-bit PNG + 8-bit 预览 |
| **毫米波雷达** | 弓形布束 → 面元物理光学 (PO) → 雷达方程 | 距离-方位图 + 检出点 + 3D 框 |
| **激光雷达 LiDAR** | 脉冲飞行时间 + 激光雷达链路方程 | 距离图 + 强度图 + PLY 点云 |
| **标注** | 掩膜 / 投影 / 点云 | COCO JSON、雷达点云 JSON、PLY |

关键点：**所有模态在同一帧里共享同一份三维世界与同一套天气状态**，
所以可以做跨模态配准、融合、以及"同一目标在不同传感器下长什么样"的对照研究。

---

## 2. 快速开始

```powershell
# 自检（20 项，含 NUC 固定性、波段发射率、海面粗糙度、MTI 门控）
set PYTHONPATH=%CD%
env\Scripts\python.exe -m simkit selftest

# 生成一套三模态数据集：6 帧、三天气、光学 + 双波段红外 + 雷达
env\Scripts\python.exe -m simkit gen ^
  -c scenes\final_air_uav.json ^
  -o out\final_air -n 6 -s ir,radar,optical

# 加激光雷达
env\Scripts\python.exe -m simkit gen ^
  -c scenes\final_air_uav.json ^
  -o out\final_air -n 6 -s ir,radar,lidar,optical

# 三个场景合并成一份 COCO
env\Scripts\python.exe tools\merge_dataset.py out\dataset out\final_air out\final_ground out\final_sea

# 导出网站数据，然后本地起个静态服务打开 index.html
env\Scripts\python.exe tools\export_web.py . out
node tools\serve_web.js          # http://127.0.0.1:18080
```

网站也可以直接双击 `index.html` 用 `file://` 打开（`fetch` 会被浏览器挡掉，
所以这时改用本地静态服务；GitHub Pages 上是正常的）。

没有 Blender 时把 `optical` 从 `-s` 里去掉即可，其余模态纯 numpy，秒级出图。

---

## 3. 仓库结构

```
simkit/                仿真内核（纯 numpy）
  scene.py             JSON 配置 → 三角面世界；prepare_mesh() 是网格准备唯一入口
  mesh.py              OBJ 载入、顶点聚类抽稀、海面/平面生成
  bvh.py               向量化射线求交（雷达 / 激光 / 遮挡共用）
  thermal.py           逐面稳态能量平衡（太阳 + 天空 + 对流 + 内部热源）
  ir.py                Planck 波段积分、大气透过率、PSF、NETD、NUC/FPN
  radar.py             面元 PO 散射、雷达方程、MTI 门控、距离-多普勒合成
  lidar.py             脉冲 TOF、链路方程、PLY 导出
  weather.py           8 维天气状态机 → 三个模态的物理量
  optical.py           Blender 调用封装
  dataset.py           帧循环、域随机化、COCO 标注
  selftest.py          20 项物理自检
blender/render_optical.py   Blender 端渲染脚本
scenes/                 场景配置（空中 / 地面 / 海面）
assets/                 3D 模型与材质
tools/                  数据集合并、预览拼接、Web 导出、雷达预算表、站点探针
index.html app.js app.css  静态网站（无构建步骤、零依赖，GitHub Pages 直接托管）
docs/                   标定与验证记录
```

---

## 4. 物理模型要点

### 4.1 天气是一个状态机

8 个标量 `cloud_cover, visibility_m, rain_mmh, wind_ms, air_temp_c, rh, sun_elev_deg, sea_state`
经 `derive()` 变成每个模态各自消费的物理量。雨在光学、红外、雷达里是**同一场雨**。

- 能见度 → Koschmieder 消光 `β = 3.912 / V_km`
- 太阳 → `G_dir = 1361 · 0.75^(m^0.678) · sin(el)`，按云量做直射/散射分离
- 天空长波温度 → Swinbank 晴空值按云量向云底温度插值
- 雷达雨衰 → ITU-R P.838-3 `γ = k·R^α`，双程累计
- 激光消光 → 由 550 nm 的 Koschmieder 值按 `λ^-1.3` 折算，再叠加雨滴项

### 4.2 红外

逐面稳态热平衡：`α_solar·G + ε·σ·T_sky⁴·F_sky + h·(T_air − T) + q_int = ε·σ·T⁴`。
解出每面温度后做 Planck 波段积分，乘大气透过率，卷积高斯 PSF，加 NETD 噪声。

**NUC/FPN 在量化到 16-bit 之后施加**，种子取自配置而非帧号，所以固定图案噪声在整个数据集里是同一张，
符合真实焦平面阵列的行为。

### 4.3 雷达

弓形栅格逐条射线求交 → 每个面元按物理光学近似给 RCS：

`σ = 4πA²cos²θ/λ² · ρ² · coh`，`coh` 是 Kirchhoff 粗糙度衰减。

再走标准雷达方程 `P_r = P_t G² λ² σ / ((4π)³ R⁴ L)`，
噪声 `k·T₀·F·B·L`，处理增益 `10log₁₀(n_chirps)`。

六个频段预置：X / Ku / K / Ka / W / W94，各带自己的天线波束宽度与雨衰系数。

### 4.4 激光雷达

905 nm 脉冲，方位/俯仰栅格逐条求交。返回光子数按链路方程：

`N = E_t · (ρ/π) · (A_r/R²) · cos(inc) · exp(−2βR) · η / E_ph`

唯一一个仪器常数——"反射率 10% 目标的作用距离"——用来标定噪声底，
这样灵敏度是一个可核对的数字，而大气、反射率、距离仍然跟着物理走。

---

## 5. 成品结果

三场景 × 6 帧（晴/阴/雨各 2 帧）× 五模态：

| 场景 | 目标 | 距离 | 雷达 | 光学框 | 雷达检测 | 链路余量 |
|---|---|---|---|---|---|---|
| 空中 | Global Hawk（39.9 m 翼展） | 1.5 km | Ka 35 GHz | 310×174 px | 17–27 点/帧，SNR 25–29 dB | +17.8 dB 晴 / +13.2 dB 雨 |
| 地面 | 巡视器 + 路障/护栏/灌木 | 120 m | W 77 GHz | 102×64 px | 22–27 点/帧，SNR 16–17 dB | +5.5 dB 晴 / +4.2 dB 雨 |
| 海面 | 帆船（真实模型） | 150 m | Ka 35 GHz | 221×231 px | 123–172 点/帧，SNR 63–65 dB | +53.4 dB 晴 / +52.9 dB 雨 |

「链路余量」= 该帧实测 SNR − 检测门限，可以直接对账：空中 11 + 17.8 ≈ 28.8 dB，
和上面的 25–29 dB 是同一个数。余量为负就说明目标已出作用距离，出 0 点是物理结果而不是 bug。

合并后的 COCO：optical / ir_mwir / ir_lwir 各 18 图 18 框，类别 `uav / vehicle / boat`。

---

## 6. 在线看板

仓库根目录就是网站本体（`index.html` + `app.js` + `app.css` + `data/`），
推上 GitHub Pages 之后直接可访问，**零构建、零依赖、零 CDN**，
断网环境下把整个仓库拷走、起一个静态服务就能用。

六个页签，前五个不只是"看"，都能动手：

| 页签 | 做什么 |
|---|---|
| **总览** | 三场景 × 6 帧，7 种模态（可见光 / MWIR / LWIR / 雷达距离-角度 / 激光距离 / 激光强度）切换，带检测框叠层与自动播放 |
| **对照** | 三种天气并排看同一目标，比较雨对可见光、红外、雷达各自的打击 |
| **激光雷达** | PLY 点云在浏览器里边实时渲染，可拖动旋转、滚轮缩放；显示命中率与量程 |
| **雷达** | 左边是雷达距离-角度原始图，右边是从检出点重建的俯视极坐标图；下方给出**链路预算余量**（天线增益、RCS、门限、发射功率、积累数，晴/阴/雨三档最大距离与余量） |
| **参数实验** | 浏览器里实时算雷达方程、ITU-R P.838 雨衰、Koschmieder 能见度、激光雷达链路。拖滑块看换波段 / 下雨 / 拉远到底损失多少 |
| **关于** | 物理模型、验证数字、已知局限 |

"参数实验"页的常数不是手抄的：`tools/export_web.py` 从 `simkit/radar.py`
直接导出 `data/radar_bands.json`（每波段天线增益由真实波束宽度算出），
所以网页上的数字和内核永远一致，改内核不会让网站偷偷说假话。

---

## 7. 验证过什么

- `python -m simkit selftest`：20 项全 PASS
- 光学亮度对齐 11 张实拍无人机照片（天空 luma 135/150/161/191/226，目标比天空暗 39–52 级）
- 红外 NUC 固定图案相关 0.962；LWIR 目标对比度 +59…+81 灰阶
- 雷达雨衰实测：2 km Ka 晴 35.1 dB → 8 mm/h 雨 28.8 dB（Δ6.3 dB）
- 各波段作用距离表（σ = −10 dBsm，10 dBm 发射、32 chirp、11 dB 阈值，天线增益按各波段
  自己的波束宽度由 `4π·η/(θ_az·θ_el)` 算出，而不是所有波段套一个 30 dBi）：

| 雨强 | X | Ku | K | Ka | W | W94 |
|---|---|---|---|---|---|---|
| 0 mm/h | 1.7 km | 1.9 km | 1.9 km | 2.4 km | 0.3 km | 0.5 km |
| 4 mm/h | 1.7 | 1.8 | 1.7 | 2.0 | 0.3 | 0.4 |
| 16 mm/h | 1.6 | 1.6 | 1.4 | 1.4 | 0.3 | 0.3 |
| 64 mm/h | 1.3 | 1.1 | 0.9 | 0.8 | 0.2 | 0.2 |

  表由 `tools/radar_budget_table.py` 生成，网站「参数实验」页用的是同一批常数
  （`data/radar_bands.json` 直接由内核导出，浏览器不另抄一份）。

---

## 8. 已知局限

1. **雷达 RCS 是几何 PO 估计 + `rcs_db` 覆盖值**，不是实测。无人机 +1.5 dBsm 是合理占位值。
   换一个 RCS 假设，作用距离会显著移动（表在 §7），但整套流程不用改。
2. **SAR 成像模式尚未实现**（仓库名中的 SAR 目前由毫米波雷达的面元散射 + 雷达方程承担，
   提供的是 SAR 成像所依赖的散射中心与衰减模型；条带式成像在路线图上）。
3. **海面是六正弦叠加**，无白冠、无破碎波；海气温差未建模。
4. **MWIR / LWIR 预览图高度相关**（相关系数 0.995+），因为同一表面温度主导两者；
   波段差异在 16-bit 原始计数里（平均 46–63 counts ≈ 175–240 mK）。
5. **天气只有晴/阴/雨三档**，缺雾、雪、沙尘。
6. 光学只标定了天空亮度，地面与阴影未单独标定。
7. 单帧耗时：红外+雷达 ≈2.2 s，光学 ≈4.3 s（CPU）。大批量建议并发跑多个进程。

本项目是**仿真工具**，输出用于算法训练与验证；数值为工程近似，不替代实测标定。

---

## 9. 许可与素材

代码：MIT。

3D 模型来自 **NASA 3D Resources**（公有领域）与 **Poly Haven**（CC0），
详见 `assets/CREDITS.md`。所有素材均可商用，无需署名（仍在此致谢）。