# 下一步计划

> 2026-11-01 真实性 v3 已补齐：源材质/UV、共享重复面清理、前向修正、剖面大气、局部热源、
> RGB 光电子/曝光、实测 RD 峰检测和独立实例/深度检查。见 [v3 记录](docs/真实性升级_v3.md)。
> 下面为此前计划，保留历史背景；剩余重点为真实设备参数/样图校准、时域 IR/RD、热惯性、
> 独立模型与场景分组的训练/测试验证。没有真实参考数据时不能用“图像更清楚”代替真实性验收。

当前状态：四类观测几何的无人机导引视角场景（高空空中目标 / 低空上升接近 / 城市遮挡 / 低空下视杂物）
已跑通全部物理模态与 YOLO 导出；跨模态配准由同一套三维投影保证，导出时逐帧自检。
本文件列出**接下来要做什么、为什么、怎么验收**。

顺序按"影响 × 成本"排：先做会污染训练标签的，再做会限制规模的，最后做体验类的。

---

## P0 · 标签正确性（在做训练集之前必须解决）

### P0-1 未标注的强散射体

**问题**：低空下视场景里有几何与目标完全相同、但 `label:false` 的诱饵车
（`decoy_rover`，pos = 38, 430, 0），另外 `fence_2` 这类围栏在雷达/SAR 上的回波
比目标还强。导出成 YOLO 后它们变成"同类不标"——检测器检出它们会被扣分。

**为什么必须现在做**：这是唯一会**直接教错模型**的问题，其它问题只是"样本少"。

**方案（三选一，默认 A）**

| 方案 | 做法 | 代价 |
|---|---|---|
| A | 给诱饵车单独一类 `decoy` | 场景配置一行；导出器自动多一个类 |
| B | 把诱饵车也标成 `vehicle` | 一行；但模型学不到"这是假的" |
| C | 从训练导出里剔除该场景 | 一行；少一个场景 |

**验收**：`tools/export_yolo.py` 导出的每个类别都有正样本；场景里不再出现
"与目标同型却无标注"的物体。

### P0-2 宽视场框只有 1.3–3.4 px

**问题**：宽视场（18°）的目标在 640×480 上物理尺寸就是 1–3 px。中位数 3.1×1.7 px，
最小宽 1.3 px，标准 YOLO 基本学不动。

**方案**

- A（推荐先评估）：宽视场分辨率提到 1280×960，框变成 2.6×1.4 px——**仍然不够**；
  要 6 px 需要 2560×1920 级，渲染成本 ×4。
- B：保留现状，但在 `export.json` 里显式标记 `search_stage_only: true`，
  训练脚本默认跳过这三套。
- C：给宽视场加**切图（tiling）**导出：把 640×480 切成 4×4 的 320×480 小窗，
  目标在切片里放大 4 倍。训练时用切片，评估时用整图。

**验收**：宽视场那三套要么框 ≥6 px，要么在导出清单里被明确标记为不可训练。

### P0-3 SAR 序列准静态

**问题**：SAR 每帧把波束对准目标（否则平台 180 m/s 的位移会把目标推出方位窗），
代价是目标在 20 帧里只漂 2–3 px，帧间差异主要是斑点噪声，作为检测样本信息量极低。

**方案**：加宽 `az_extent_m`（当前 400 m），让目标随平台运动自然扫过成像区，
不再逐帧重新对准。代价：方位分辨率随孔径变化，需要重算 ρ_a。

**验收**：目标在 20 帧里的横向位移 ≥20 px，且 SAR 目标对比度不低于当前值。

---

## P1 · 规模与覆盖

### P1-1 全量四场景 × 240 帧重生成

当前仓库里的 `out/` 只有 20 帧/场景的验证样本（`seeker_air2air_high` /
`seeker_surface2air` / `seeker_air2ground`），城市遮挡场景尚未重生成。

```powershell
python tools\build_seeker.py -s optical,ir,radar,lidar,sar --jobs 5
python tools\export_web.py . out
python tools\export_yolo.py
```

**验收**：`tools/check_seeker.py` 四场景全 PASS；遮挡场景出框率仍为 75%
（每段前 30 帧被楼体挡住）；导出后 `datasets/yolo` 三个 split 都有样本。

### P1-2 类别与片段均衡

当前 20 帧样本里 `uav` 与 `vehicle` 各占一半，但每个场景只有 1–2 个片段。
全量后应有 4 场景 × 2 片段 = 8 段，导出器按片段切 train/val。

**验收**：`export.json` 里每类正样本数 ≥ 200；train/val 的类别比例差 < 10%。

### P1-3 雷达链路预算表

现在雷达参数是手工调出来的（40 dBm + 2°×6° 波束）。需要一张自动生成的表：
不同 RCS × 不同斜距 → SNR 与检出概率。

**验收**：脚本输出 CSV，覆盖 0.1–10 m² × 1–12 km，误差与解析式 <1 dB。

---

## P2 · 工程化

### P2-1 导出前的"同批次"硬闸门

这一轮踩过坑：改一次参数就重跑，目录里混进两套参数的帧，导出会把两种噪声风格
烤进同一个数据集。导出器必须先校验每个 run 的文件属于同一批次
（mtime 跨度 ≤ 单批耗时 × 2，且写入已停止 ≥120 s），不满足就拒绝导出。

**验收**：故意造一个混批目录，导出器报错退出而不是静默通过。

### P2-2 CI

**现状：CI 未启用，push 后没有任何自动闸门。** `.github/workflows/selftest.yml`
只存在于本机磁盘，被 `.gitignore` 的 `.github/workflows/` 一行排除：推送 workflow
路径需要带 `workflow` scope 的凭据，而当前 OAuth token 的 scope 实测为
`gist, read:org, repo`，GitHub 会拒绝。质量闸门目前只有本地两条命令：
`python -m simkit selftest`（50 项）与 `python -m unittest discover -s tests`（458 项）。

解除办法二选一：换带 `workflow` scope 的 token 后删掉 `.gitignore` 那一行并提交；
或用 GitHub 网页编辑器直接新建该文件（网页会话不受 OAuth scope 限制），本地仍需
解除 ignore 才能跟踪后续修改。

入库后 workflow 应跑：`python -m simkit selftest` + `python -m unittest discover -s tests`
+ YOLO 格式校验（标签字段/范围/图文配对/切分无泄漏）。其中前两项已有；YOLO 侧
`tools/export_yolo.py` 的自检目前只覆盖跨模态标签一致性与图文配对，
"切分无泄漏"只由按片段切分的构造保证，还缺一个独立校验步骤。

### P2-3 跨模态像素级对齐的可观测量

现在的"配准自检"比的是**标签**（同一套投影导出，必然相同），属于回归防护，
不等于证明两张图对齐。要证明像素对齐，需要一个可观测指标：
例如用互相关估计红外与可见光的同一目标质心差，目标是中位 < 2 px。

### P2-4 多智能体协作的写冲突防护

并行 agent 同时写 `out/` 会造成混批（已发生）。约定：
每个 agent 拥有独立目录，合并前由主控校验批次；代码文件按模块划分所有权。

### P2-5 站点与构建素材分离发布

**问题**：站点是仓库根目录直接发布的，所以 254.3 MB 的构建输入
（`assets/backdrops`、`assets/models`、`assets/refs`）跟着一起上了
Pages，而浏览器只会取 28.8 MB（`data/` + `assets/hero` + `assets/figures`）。
这与 README §6 自述的体积纪律相矛盾。

**为什么排在最后**：不构成故障——总量 318.2 MB 远低于 Pages 1 GB 上限，
最大单文件 16.9 MB 也没触发 100 MB 限制，站点打开速度只取决于它真正取的 28.8 MB。
代价是首次部署慢与仓库膨胀。

**做法**：把站点产物发布到独立的 `gh-pages` 分支（只含 `index.html`、`app.js`、
`app.css`、`.nojekyll`、`data/`、`assets/hero`、`assets/figures`），主分支保留构建输入；
`tools/export_web.py` 增加一个"导出到发布分支工作树"的目标参数。
或退一步：把四个素材目录迁到 Git LFS。

**验收**：Pages 部署体积 ≤ 40 MB；`node tools/probe_web.js <线上 URL>` 仍 11 个页签
全开、0 破图；主分支上 `assets/models/rover.obj` 之类的大文件不再出现在发布产物里。

---

## 不做的（写下来免得反复讨论）

- **MATLAB / GPU 依赖**：整套仿真是纯 numpy + 无头 Blender，保持不变。
- **真实雷达数据配准**：仿真数据没有真实传感器的时间戳与姿态误差链，
  做"真值对齐"会引入无法量化的偏移。
- **宽视场做成可训练主数据集**：见 P0-2 的结论，物理尺寸决定了它的定位。

---

## 里程碑

| 阶段 | 内容 | 验收 |
|---|---|---|
| M1 | P0-1、P0-2 | 导出清单里没有"同类不标"，不可训练的套件被显式标记 |
| M2 | P1-1、P1-2 | 四场景 × 240 帧全量出站，站点与 YOLO 同步更新 |
| M3 | P0-3、P1-3 | SAR 序列有真实位移；链路预算表可复现 |
| M4 | P2-1、P2-2、P2-3 | 导出带批次闸门；CI 入库且首跑绿（需 workflow scope 凭据）；像素对齐指标 <2 px |
| M5 | P2-5 | Pages 部署体积 ≤ 40 MB，站点自检仍全绿 |

---

## 循环计划：多传感器仿真数据集的最后落地

上面 P0/P1/P2 说的是"要做什么"，这一节说的是"按什么节奏反复做直到收口"。
每一轮都跑同一套九步，**任一闸门红就停在那一步：不发布、不提交**，修完从第 3 步
重跑；九步全绿才允许进入下一轮。

### 每轮九步

| # | 步骤 | 命令 | 通过判据 |
|---|---|---|---|
| 1 | 起点干净 | `git status --short`；`git rev-list --left-right --count HEAD...origin/main` | 无未提交改动；不落后于 origin/main |
| 2 | 基线绿 | `python -m simkit selftest`；`python -m unittest discover -s tests`（**设与不设 PYTHONPATH 各跑一次**） | 50 项 ALL PASS；两次都是 `Ran 458 tests ... OK` |
| 3 | 单批次生成 | `python tools\build_seeker.py --dest out\<本轮批次> -n 240` | 只写本轮目录；四个场景齐全 |
| 4 | 批次闸门 | 待实现（P2-1）：run 内图像 mtime 跨度 ≤ 单批耗时 × 2，且写入已静默 ≥ 120 s | 混批要退出码 2，不能静默导出 |
| 5 | 物理闸门 | `python -m tools.validation_check out\<批次>\<场景> -s <模态>`；`python tools\check_seeker.py out\<批次>\*` | `passed: true`；无 hard failure |
| 6 | 导出闸门 | `python tools\export_yolo.py --runs out\<批次>` | 退出 0；`skipped_unnamed = 0`；跨模态标签最大偏差 ≤ 0.51 px |
| 7 | 站点发布 | `python tools\export_web.py . out`；`python tools\opt_images.py` | manifest 的场景数与帧数与本轮一致 |
| 8 | 站点自检 | `node tools\check_web.js`；`node tools\check_data.js`；`node tools\probe_web.js` | 11 页签零错配；`DATA SHAPE OK`；0 破图、0 JS 错误 |
| 9 | 线上核对 + 留痕 | 比对 Pages 与本地 `data/manifest.json` 的 sha256；一条 commit 写清本轮证据数字 | 字节一致；本文件对应条目改为已达标 |

第 2 步要跑两遍是有原因的：后端子进程继承的是**环境变量**而不是父进程的 `sys.path`，
只测设了 PYTHONPATH 的那一次，就会漏掉 CI 那条路径上的失败——这个坑已经踩过
（`simkit/external.py` 的 `child_env()` 就是为此加的），两遍都跑才守得住它。

### 轮次分配

| 轮 | 内容 | 收口条目 | 里程碑 |
|---|---|---|---|
| 1 | **已完成**（commit `ceb483e`）：可复现性与演示链路修复——后端子进程 PYTHONPATH、导出器遇 list 形标注崩溃与拼接崩溃、四处硬编码绝对路径、校验闸门的模态作用域、`--dest` 只生效一半、README 环境引导与过时数字、离线部署手册的依赖错误、og:image 404、根目录垃圾文件 | — | — |
| 2 | 诱饵车单独成类（场景里给 `cls:"decoy"`，导出器的 `class_map()` 会自动多一类）；宽视场套件写 `search_stage_only` 并在 `data_*_wide.yaml` 首行注明 | P0-1、P0-2 | M1 |
| 3 | 四场景 × 240 帧全量重生成，补上至今缺失的 `seeker_air2air_occluded`；类别均衡到每类 ≥ 200 框 | P1-1、P1-2 | M2 |
| 4 | SAR 停止逐帧重新对准（加 `sar.follow_target` 开关，默认不再重写 anchor、不再把 `_frame_t` 置零），并放宽 `az_extent_m`，使帧间位移 ≥ 20 px；链路预算表改成 RCS × 距离二维网格、输出 SNR 与 Pd 并落 CSV | P0-3、P1-3 | M3 |
| 5 | 导出器加批次闸门；workflow 入库并首跑绿；图像级配准指标（互相关质心差中位 < 2 px）接进 seeker 链 | P2-1、P2-2、P2-3 | M4 |
| 6 | 站点与构建素材分离发布，Pages 体积降到 ≤ 40 MB | P2-5 | M5 |

**轮次之间有依赖，顺序不能随便换**：轮 2 的诱饵成类之后必须重生成，否则旧产物里
仍是"同类不标"；轮 4 的 SAR 改动同样要求重生成。所以**轮 3 的全量生成不要提前跑**——
更省的排法是把轮 2 和轮 4 的代码改动都做完，再用轮 3 跑唯一一次 240 帧全量。
光学渲染是这条链上最贵的一步，"改一点就重跑全量"是最贵的错误。

### 停止条件

轮 6 结束且 M1–M5 全部达标即收口。任何一轮里第 4–8 步出红灯，本轮不算完成、
不计入轮次，修完从第 3 步重跑。

### 每轮的固定开销

第 1、2、8 步是纯校验，合计约 3 分钟；第 3 步（240 帧含光学）是唯一的长任务。
第 9 步的线上核对只在真的推送之后才需要跑。
---

## 客户视角改进清单（2026-11-06 评估）

以真实客户身份验收后的结论：**两个投诉都能修，而且都便宜**；真正的短板不在能力缺失，
在于已建好的能力没有接到出口上。

### 一、市场对标

最近同类是 DIRSIG（免费 Cornerstone 版、EAR-99、无加密狗；但 DIRSIG5 当前**无 SAR、无偏振**，
SAR 只在旧版 DIRSIG4）、OKTAL-SE SE-Workbench（EO/EOS/AEO/RF/SAR/GNSS/声学，SAR 走射线追踪 RCS）、
Ansys AVxcelerate + STK EOIR（24 传感器 / 36 波段，GUI 可配探测器阵列）、MuSES（红外特征）。
开源侧 CARLA（15k★，活跃）、Isaac Sim/Replicator（RTX 强制）、BlenderProc（3.8k★，最接近我们的光学链、
自带 COCO 写出）、Mitsuba 3、Sionna RT（GPU-only）；AirSim 已停（后继 Colosseum）、LGSVL 2022 停、
Unity Perception 2024-11 停。直接先例是 SAMPLE（AFRL 合成+实测配对 SAR）、MSTAR，
以及 SynDroneVision（WACV 2025，游戏引擎、仅 RGB）——后者是最接近的已发表先例。

**领先是结构性的，不是保真度**：没有开源工具在一个离线 CPU 栈里同时出五模态逐像素配准帧 +
自动 COCO/YOLO；反无人机与窄视场机载导引 + 电磁对抗这个细分在仿真侧是空白（飞行类数据集全是实测）；
离线 / 无 GPU / 无授权的运行时正好卡在采购敏感场景。但在辐射与电磁保真度上，我们落后于每一个防务 COTS。
另：全部 COTS 报价均为询价制，未取得；VBS4 属 BISim/BAE，不是 Presagis（CAE 分拆）。

### 二、投诉一「图像仿真不真」——根因是开关没打开

1. **相机 ISP 的四个镜头/压缩阶段默认关闭**。`simkit/camera.py` 实现了完整六阶段
   （vignetting / chromatic_aberration / diffraction / flare / sensor / jpeg），链路已接
   （`render_optical.py:53` 算 has_effect、`:1031` 调 apply），有专测 `tests/test_camera.py`。
   实测更正：七个 run 的 active stages 是 `['diffraction','sensor']`——即模块默认值已开了衍射 OTF 与
   探测器噪声两级，且这两级**测得是对的**（最平图块 std 1.19–1.33 DN、八帧时域 std 1.03–1.09 DN，
   与文档给的 ISO-100 目标 1.5–1.7 DN 相符）。默认关闭的是渐晕、横向色差、眩光、JPEG 四级；
   实测通道相关 R-G 高达 0.9667–0.9993，而真实 f/2.8–f/5.6 镜头有 0.5–1.5 px 横向色差与 20–40% 角部失光。
   同时 32 个场景**无一处**写 `optical.camera` 块，全部走默认值，无任何场景级调校。
   模块文档串自己写着：「在干净版本上训练的检测器会学到天空是平场、每个边缘恰好锐利、三通道相互独立
   ——这三条对真实相机全不成立」。**这一级修复 = 加配置块，零代码、零新依赖。**
2. **材质链同样没启用**：`import_materials` 与 `optical_material_source` 在 32 个场景里 0 个开启
   （`material_helpers.py:128` 默认 False），而 `blender/material_helpers.py` 已支持 OBJ/MTL 逐面材质
   （map_Kd / 法线 / 粗糙度 / 金属度）与 glTF PBR 节点复用，`simkit/mesh.py` 已保留 UV、源材质名与分组。
   带真贴图的 GLB **已入库**：`assets/aviation/uav.glb`(10.1 MB)、`assets/aviation/airliner.glb`(29.1 MB)。
3. **但走 OBJ 拿不到贴图**：`global_hawk.mtl` / `uav.mtl` / `airliner.mtl` / `ingenuity.mtl` / `rover.mtl`
   是纯颜色、无任何 map_ 槽；有贴图的四个 .mtl 其 map_Kd 全指向不存在的 `C:/xxx_diff_2k.jpg`
   （仓库内找不到，本机 C 盘根目录也没有）→ 即便打开 `import_materials` 也只会记 missing_textures 后回退。
   `docs/experiments/2026-11-01_materials.md` 自己写明「GLB 本身带纹理，但既有 OBJ 转换输出的 MTL 不含这些贴图」。
   **正解是让 `optical_material_source` 指向 GLB，不是去修 OBJ/MTL。**
4. 目标还被 `decimate: 12000` 砍到 1.2 万面，几何细节同步丢失。
5. **红外侧项目已有权威台账，不必重新发现**：`docs/IR_经验模型与改进方案.md` §6 分级 A1–A4 / B1–B5 / C1–C6，
   §0 给了 6 条 MVP。A 级是会系统性污染训练集的物理错误——A1 完全没有路径辐射项（LWIR 绝对亮温偏冷
   10–40 K 且随距离单调漂移，模型会学到「远=冷」）；A2 宽带天空温度直接代入波段普朗克（MWIR 天空辐射高
   3–8 倍、虚高约 50 K）；A3 常数消光 0.10/0.06 每公里（10 km 处透过率误差 2–4 倍）；A4 Swinbank 乘
   (1−0.25RH)（晴空天空温度偏低 25–35 K、地表系统性偏冷 5–7 K）。B1 太阳辐照双重计数（偏高 2–5 倍）、
   B3 无漫射太阳（背阴面白天纯黑）。C4 = 16 bit 编码线性于温度而非辐射/电子数；C6 = 无大气湍流与
   衍射/像差 MTF。
6. 产物侧实测（原生位深、未降位）：机载导引光学 768×576 全图 luma std 仅 2.3、只用 75 个灰度级；
   radar RD 1024×64、中位数 0（近全黑）、std 6.9；IR 为 16 位 I;16（ir_mwir 5186 个灰度级、天空角 std 259），
   **数据本身不空**，但任何用 convert(L) 的下游会静默压成 2 级剪影——`docs/YOLO导出.md:38-39` 早已写明
   该陷阱并因此让 YOLO 只用 8 位预览，评估时不要重复踩。
以下六条来自独立取证（`_tmp/batchE_realism/REPORT_F.md`，65 KB，18 条缺陷全部带 file:line 与实测数），
严重度均高于上面第 1–6 条，且**前十条的修复都不需要任何新依赖**——所需的物理模型早已写好躺在仓库里没用：
`optical_sensor.py`、`atmosphere.py`、`preview_span_c`、`region_overrides`、`configure_exposure`、
`view_transform_available`、`png.write_gray16`。

7. **红外图其实是一张可见光照片**（演示阻断，最伤）。`ir.py:853` 把 `plate` 直接当 float64 贴进
   img/counts/preview/temperature 的空洞像素，而 `out/demo_ground_plate/_render_cfg.json` 把
   optical.sky_image、ground.backdrop 与 ir.background_image **同时**指向同一张
   `assets/backdrops/city/06_Drone_view_of_roundabout_Unsplash.jpg`；`_background_temperature_k()
   (ir.py:454-475)` 把 8 位亮度**线性**映射成温度。实测：红外预览 256 个灰度级里有 255 个（99.6%）
   也出现在源 JPEG 中；HF 能量 ir_preview 0.354 vs JPEG 0.366 vs 光学渲染 0.171。客户看到的是
   **斑马线白漆成了 MWIR 帧里最热的东西**——真实 LWIR 里同温白漆与黑沥青只差 2–4 K，斑马线几乎不可见。
   修法：(a) 一次代码搬移——把 `_composite_background`(`ir.py:1034`) 挪到 NETD/模糊/NUC 段
   （`ir.py:961-1015`）**之前**，让底板走同一条探测器链；(b) 约 60 行，用 `ir.py:375` 已有的普朗克反演
   把底板反射率转成温度，以 kappa≈0.15 与解算出的地面温度混合，而不是 1.0 直接覆盖。
8. **8 位红外预览零噪声，且与 16 位产物不一致**（演示阻断）。预览在 `ir.py:1024` 由 img（辐射）重建，
   而整个 NUC 段（`ir.py:998-1015`：PRNU/DSNU/列 FPN/坏像素）**只写 counts**。显示窗取
   `np.percentile(t_img,[1.0,99.5])`（`ir.py:407`，因为没有场景设 preview_span_c）。实测
   hero_ground_vehicle 的 MWIR 窗口 [252.1,311.4] K = **每灰度级 232.5 mK**，于是宣称的 30 mK NETD
   只占 0.13 个灰度级，8 位高通 MAD **恰好 0.000 DN**，而 16 位产物是 825.7 mK；lwir 与 final_ground
   同为 0.000 DN。`ir.py:393-400` 早已写明该失效并给了修法，无场景采用。
9. **旗舰窄视场是一大片无特征灰雾**（演示阻断）。`out/seeker_air2ground/images/optical/000000.png`
   768×576 实测对比度 p99−p1 = **255 级里只有 9.24 DN**，44.2 万像素里仅 1309 个不同 RGB 三元组（0.30%），
   最平图块 std 0.73 DN。同一 run 的宽视场有 132 DN 对比度——**高 14 倍**，所以问题特定于 2.2° 窄通道。
   成因：地面没挂底板 → 落到 `add_ground_detail()` 的程序化 Voronoi（`render_optical.py:225-270`，
   ground_scale_m=6.0、对比 ±0.16），再被 `auto_exposure`（目标平均亮度 0.48，`:876`）压平，且无消光（见 12）。
   修法纯配置：像 demo_ground_plate 那样挂地面底板、ground_scale_m 调到 1.5–3、ground_contrast 0.5、
   窄通道关掉 auto_exposure。该套机制已在三个 run 里验证可用。
10. **目标等温，真正的热解算被丢弃**（不合理，但客户一眼就看出来）。`thermal.py:183-292` 是一个真正的
   牛顿迭代辐射/对流平衡（太阳直射+漫射、BVH 投影、天空/地面视角系数、含气动加热的强制对流、雨湿、
   树冠降温、60 次迭代）——做得很好。但 `thermal.py:284-290` 随后用逐对象 temp_c 与地面 temp_c **覆盖**它，
   而 `temp_c` 出现在 **32 个场景中的 30 个**。例：`scenes/hero_ground_vehicle.json:57` 给 vehicle_rover
   设 temp_c 39 → 整车 312.15 K 等温，加上 `surface_temperature_texture()` 的 texture_sigma_c 0.6
   总温差只有 1.2 K。真实行驶的 rover：发动机舱 60–90 ℃、排气 150–300 ℃、轮胎 40–60 ℃、
   向阳顶面约 45 ℃、背阴面约 25 ℃，跨度 40–70 K。客户看到的是一个饱和的无特征白椭圆，
   太阳在 40° 且光学帧里影子清晰可见，红外却没有任何向阳/背阴不对称。
   修法纯配置：删掉 temp_c 让 solve() 生效；或约 30 行走 `thermal.py:216 region_overrides`
   （已在 `:289-290` 应用）给发动机/排气/轮胎/风挡分区，`simkit/skin.py` 已建模表面分区；
   或把 temp_c 当**偏置**而非覆盖。**注意：不要用 `tools/make_seeker_scenes.py` 重生成场景**（会覆盖手工补丁），
   直接改 JSON。
11. **光学波段单独在真空里渲染**（不合理，且是最便宜的高可见度收益）。`atmosphere.path_transmittance()`
   （`atmosphere.py:97`，消光在 `:13`）被 IR（`ir.py:324` 逐像素 tau 与路径辐射）、SAR（`sar.py:448-454`
   双程 radar_att_db_km）、LiDAR（`lidar.py:103` 双程 beta_lidar_per_km）调用，但可见光只在
   `optical_sensor.py:161` 调用，而它被 `optical.sensor.enabled` 门控——实测七个 run 全部 sensor_model=False。
   `render_optical.py` 里 volume/fog/Scatter/absorption **零命中**，Cycles world 无体积散射节点。
   实测：数公里外的山脊与相机前 20 m 的草**对比度和暗度完全相同**，visibility_m 20000 下没有任何
   空气透视或地平线霾。与第 12 条合起来是最强的「游戏渲染」观感来源。
   修法约 25 行纯 numpy、**无需重渲染**：深度 EXR 已经产出（`use_pass_z`，`render_optical.py:834`，
   写出于 `:839-847`）且已被载入 float32（`:959-965`），只是从未被使用；在 `camera.apply()` 之前的
   **线性光**里做 L = L·tau(d) + L_air·(1−tau(d))，tau = exp(−k·d)、k = 3.912/visibility_m（Koschmieder），
    airlight 颜色用已有的 sky_zenith_rgb。
12. **光学探测器链跑在 8 位色调映射后的 PNG 上，而不是 HDR**（不合理）。`render_optical.py:56-74`
   `read_rendered_rgb8()` 读回 Blender 刚写的 PNG（colorspace Non-Color）返回 uint8，`:849-859` 把它当交付物，
   `:861-865` 在**这 8 位上**跑 `sk_camera.apply()`。view_transform 在七个 run 里全是 Standard（无高光滚降）。
   实测**每一个 run 的每一帧饱和像素占比都是 0.0000%**——真实日光相机几乎每帧都会钳位掉一些东西；
   全零钳位是 auto_exposure（目标平均亮度 0.48）加 Standard 共同保证的。hero_ground_vehicle 在
   155.5 万像素里只有 9180 个不同 RGB（0.59%）、最平图块 std 0.37 DN——**噪声底低于一个 LSB**，
   即第 8 条的失效模式出现在可见光波段。顺序在物理上也是错的（噪声与去马赛克跑在色调曲线之后），
   尽管 `camera.py:233 tone_gradient()` 正是为补偿这条曲线而写、却没被真正用上。
   修法：`optical.sensor.enabled = true` —— `optical_sensor.py` 是一条完整且有专测
   （`tests/test_optical_sensor.py`，296 行）的物理链，跑在**已经写出**（`:839-847`）、**已经载入**
   （`:959-965`）却从未被使用的 EXR HDR 上；`CAMERA_SKIP`（`:52`）已防止重复施加重叠阶段。

### 三、投诉二「没有指定大小或者像素」——引擎能做，没有入口

实测（每格真实跑过并记录退出码）：1920×1080 / 1280×720 / 640×640 / 512×512 / 4096×2160 全部精确产出，
8192×4320 在 CLI 也能出（78.1 s，工作台会以 HTTP 400 拒绝）。标注与内参**正确同步缩放**
（640×640 框宽 25.58 px → 1920×1080 76.75 px，恰好 ×3；COCO images[] 与 YOLO 归一化都对），
引擎无长宽比 / 偶数 / 2 的幂约束。缺的是入口：

- CLI 无任何尺寸或 FOV 参数（只有 -c -o -n -s --seed -q --part）；build_all / build_seeker / export_yolo
  同样没有 → 唯一办法是手改场景 JSON 的 `clip.views[].width/height`
- 工作台**后端已支持** narrow_width / narrow_height / wide_width / wide_height
  （`studio.py:900-907`，限幅 LIM dim = 64..4096），但**前端 0 处发送**（两个 app.js 与两个 index.html
  全文无这些键）；`scenegen.py:1684-1688` 与 `:1243-1244` 把 960×720 / 1280×960 写成字面量；
  validate_recipe 的 LIM 字典里根本没有 width/height 键；`_recipe_to_form`(`studio.py:312-331`) 只摊平
  narrow_fov/wide_fov，连存档 recipe 里带的尺寸也会丢；`studio.py:1447` 把布站预览硬夹成 4:3、上限 960
- 对标：CARLA(image_size_x/fov)、AirSim(每请求 width/height)、Replicator(render product resolution)、
  Gazebo(SDF)、STK EOIR(属性 GUI) **全部对用户暴露分辨率**

四个连带缺陷：

- `0x0` 静默变 480×360、exit 0、无警告 —— `dataset.py:106-107` 的 falsy-zero
  （`int(v.get(width) or view[width] or 480)`）
- 非光学模态**完全忽略**尺寸请求：SAR 实测 1920×1080 与 640×640 都返回 768×512（= n_samples×n_cross，
  `sar.py:471`）；radar RD = (n_samples//2)×n_chirps（`radar.py:697-698`）；lidar range/intensity =
  n_az×n_el（`lidar.py:154-157`）。由物理导出本身合理，但**必须显式告知并给出「要达到 W×H 该设哪些
  物理参数」的换算**，否则用户看到的就是「我设了没反应」
- `-s sar` 在场景缺 sensors.sar 时零产出，却打印「使用引擎默认参数生成」并 exit 0
  （`dataset.py:542` 的门 vs `:599-601` 的消息）
- PNG 只有 IHDR/IDAT/IEND（`png.py:13-28` 写三个 chunk），无 pHYs/tEXt/EXIF，全仓无 GSD
  → 客户拿不到像素的物理尺寸

### 四、MATLAB / PS 裁决

**MATLAB：不建议引入运行时依赖。** 先纠正一个常见误判——R2026b **支持 Python 3.13**
（MathWorks 兼容矩阵 3.10–3.14；matlab-engine-for-python@R2026b 的 setup.py
SUPPORTED_PYTHON_VERSIONS={3.10..3.14}），所以「我们的 Python 太新」不是理由。真正的拦路虎是：

- **体积**：MATLAB Runtime win64 安装包实测 4.4–4.6 GB（对 MathWorks 下载服务器做 HTTP HEAD 取
  Content-Length：R2024a U4=4428.7 MB、R2024b=4519.1、R2025a=4665.9、R2025b=4735.1）。我们整个
  vendored env 是 131.8 MB、含 Blender 的整包约 390 MB → MCR 单体约为整个交付物的 11.5 倍，
  而 `docs/离线部署.md` §8.3 给全部东西的预算是 4.7 GB（一张 DVD）
- **路径硬冲突**：MCR 要求安装路径不含任何非 ASCII 字符，而本仓库在含中文的路径下
- **交付模型冲突**：无管理员权限的可行路径是「在有权限的机器上装好 MCR → 拷文件夹 → 把
  runtime win64 加进用户 PATH」，交付树因此**不再自包含**，且供应链里必须有一次管理员安装
- 还会推翻已发布的对外承诺：README「需要 MATLAB 吗？不需要，一行都不需要」「全部算法是 Python+numpy
  重写，无 .m/.mat 依赖」，以及本文件「不做的」章节明列 MATLAB
- 若确有需要的折中：把**单个**算法用 MATLAB Coder 编成独立可执行体、经子进程调用，MCR 作为可选外部件
  而非依赖；458 项测试与 selftest 仍须能在无 MATLAB 的 CI 上跑
- **真正该引入的不是 MATLAB，是 scipy，而且理由不是画质而是「已立项却做不出来」**：本文件 P1-3
  （里程碑 M3）要求链路预算表输出「SNR 与**检出概率**」，`docs/雷达与激光雷达仿真经验公式手册.md:281`
  早已写明「最稳实现：`scipy.stats.ncx2.sf(2*V_T, 2*N, 2*N*S)`（非中心 chi2 尾概率，与 Marcum Q 完全等价）」，
  `:255` 给 gammaincinv、`:331` 给 brentq 反解——**但 scipy 不在 requirements.txt，全仓 .py 对
  ncx2/gammaincinv/marcum/brentq 零命中（实测 grep）**；`radar.py:548` 只有硬门限
  `if snr_db[i] < det_snr_db: continue`，**根本没有 P_d**，`radar.py:444` 只有 N=1 闭式，
  `radar.py:158` 自己列 limitations 承认 not adaptive CFAR。纯 numpy 手写非中心卡方尾概率在大 N /
  大非中心参数下会溢出与精度崩塌——**这是全仓库唯一 numpy 确实做不好的地方**。
  代价实测：安装后 scipy 108.5 MB + scipy.libs 19.3 MB = **127.8 MB**（下载 34.9 MB），BSD-3，
  cp313 wheel 齐全，纯增量不动任何现有数值路径 → 458 项与 selftest 全绿不变。
  许可要点：主体 BSD-3 对闭源离线交付无障碍；捆绑的 libgfortran 是 GPL-3 **WITH GCC-exception-3.1**，
  明确允许非 GPL 程序链接运行库、不污染闭源代码；但分发 wheel 时**必须随包保留 scipy 的 LICENSE.txt 全文**。
  采纳则须同步改 `docs/离线部署.md` 约 15 处（第 18 行「两个：numpy+Pillow」、第 20 行体积、
  第 36 行 env 131.8 MB、**第 65 行的体积校验断言**、第 37 行 _wheels 清单、第 226-239 行 site-packages
  清单要加 scipy.libs），否则收货方按手册自检会拿到与文档对不上的数字。
  **关键优势：第 3/15/326 行的「没有 MATLAB / 不需要 MATLAB / 无 .m 依赖」全部无需改动**
  ——这正是 scipy 方案优于 MATLAB 之处：它不需要撤回任何已发布承诺。
- 其余 Python 侧 SAR/雷达库都不合适：SarPy 2.1.1（读真实 SICD/CRSD，与本文件「不做的：真实雷达数据配准」
  冲突）、scikit-rf 2.2.0（射频网络方向，且拖进 pandas）、ISCE2（InSAR 处理框架，重且需构建）、
  RITSAR/sarsim（教学级，本仓库 SAR 覆盖面已超）
- **值得固化的正确范式**：项目已经在用 MATLAB 当「文档与校验预言机」而不引入依赖——
  `docs/雷达与激光雷达仿真经验公式手册.md:357` 写「海杂波：NRL 模型（首选，已对齐 MATLAB 官方算例到
  0.01 dB）」，`:378` 给校核「SS=2、f=30 GHz、ψ=10°、H → −36.66 dB；MATLAB `seareflectivity(2,10,30e9)`
  文档值 2.1555e-4 = −36.67 dB ✓」。建议把「MATLAB 可作校验预言机，但不进依赖、不进 CI、不进交付」
  写进本文件「不做的」章节，替换掉已过时的版本兼容理由。
- **一个后果最重的法律陷阱需单独提**：MathWorks Answers 上关于编译产物再分发的讨论中，MathWorks 员工
  Andreas Goser 指出用**大学/学术许可证**的 MATLAB Compiler SDK 编出的产物，下游商业用途有限制；
  社区高票回答明确「学术许可证只能在组织内部或与科研合作者共享，不能公开」。
  **[非官方政策原文，标未核实]** 若历史上任何一版是拿高校许可证做的，交付给商业/军工客户这条路本身要先过法务。

**PS / Adobe：结构性不可行，与价格无关。** Adobe Photoshop API 是云 Only 且需 Enterprise 合约
（前置：Adobe Developer Console 账号 + 含 Firefly Services 的有效 Enterprise 合约 + 对
ims-na1.adobelogin.com 的 two-legged OAuth），每次调用都打到 photoshop-api.adobe.io、素材走签名 URL；
不存在离线或本地无头的 Photoshop API。对气隙管线直接出局。可采纳的是「PS 能力」的开源等价物：

- **psd-tools 1.26.0**（MIT，有 cp313 win_amd64 wheel，0.7 MB）—— 接收客户提供的 .psd 分层素材
  （背景 / 纹理 / 蒙版），成本极低，可直接采纳
- **OpenCV**（opencv-python-headless，Apache-2.0，abi3 wheel cp37-abi3-win_amd64 共 41.8 MB，故 3.13 可用）
  —— 真实镜头畸变、去马赛克、色调映射、NLMeans 去噪、inpaint
- 但**优先级低于打开已有开关**：`camera.py` 已自带上述大部分能力（含真实 libjpeg 往返）。加 OpenCV 之前
  应先接上 ISP，否则是重复造已有的轮子，还要连带改 131.8 MB 的 vendored env 与离线手册的体积校验

### 五、建议执行顺序（按每单位代价的可信度排）

0. **纯配置、几小时、零依赖，先做这一批**（取证见 `_tmp/batchE_realism/REPORT_F.md` §5）：
   删掉 temp_c 让 `thermal.solve()` 真正生效（第 10 条）· `optical.motion_blur: true` +
   `optical.exposure_s: 0.004`（约 2× 渲染时间；实测 demo_air_headon 在 1/250 s 下应有 18.53 px 拖影，
   现在是 0 px）· `optical.sensor.enabled: true`（一次同时解决第 11 条真空渲染与第 12 条 8 位链，
   因为可见光消光就在 `optical_sensor.py:161`）· `ir.preview_span_c ≈ 20 K`（第 8 条）·
   SAR `display_anchor_pct: 10 → 1.5`（第 16 条：实测五个 run 的黑像素占比 0.1002–0.1006，
   正好等于这个锚点，它把指数分布的暗尾截掉了，导致实测 σ_dB 只有 Gamma 理论的 0.48–0.84 倍）·
   给机载导引场景挂地面底板（第 9 条）· 打开 vignetting / chromatic_aberration / flare / jpeg 四级
   （第 1 条）。验收：`tests/test_camera.py` 与 458 项须绿；窄视场对比度 p99−p1 应从 9.24 DN 显著抬升
2. **工作台暴露 width/height、recipe schema 补键、界面显示 IFOV/GSD 读数**（后端已有，几天工作量，
   直接消灭投诉二）；同批修 falsy-zero、给非光学模态打印尺寸换算、修 -s sar 静默零产出
3. **PNG 写 pHYs + tEXt（GSD / FOV / 位深 / 辐射标定）**，让客户拿到像素物理尺寸
4. **红外按 §0 的 6 条 MVP 修 A 级物理错误**（路径辐射、波段天空、Kruse 气溶胶消光、去掉 RH 因子、
   去掉重复 cos、噪声改逐像素 dL/dT）
5. **SAR 用 numpy 物理光学网格 RCS 升级**，并拿 MSTAR / SAMPLE 公开切片做外部校验
   —— 无 GPU、无 MATLAB 约束下最大的保真度杠杆
6. **发一份「合成训练 → 实测测试」的基准报告**（对标 Anti-UAV410 / Det-Fly / SAMPLE 实测分割，
   走 SynDroneVision 的 playbook）—— 每单位投入的可信度最高
7. 采纳 psd-tools；OpenCV 放到 ISP 接上之后再议

证据：`_tmp/batchE_market/REPORT_E.md`（约 50 KB，逐条带引用键，§6 列出全部未核实项）、
`_tmp/batchE_size/`（真实尺寸探针产物与场景副本）、`_tmp/batchE_deps/`（依赖体积实测 venv）、
`_tmp/leadE/`（原生位深测量脚本与五模态拼图）。
### 补遗：终版取证里会影响修复设计的六处细节

1. **尺寸有两条路径，修复必须同时覆盖**：32 个场景里只有 5 个走 `clip.views`（全是机载导引系列，
   768×576 + 640×480），其余 **27 个走 legacy 路径**（尺寸写在 sensors 相机块，跨度 480×360 → 1440×1080）。
   legacy 下 `_views_of`（`dataset.py:91-94`）**优先取 `sensors.optical`**，所以 `sensors.ir_camera` 里写的
   尺寸被静默忽略（`scenes/sk_air_uav.json` 声明 ir_camera 480×360，实际按 optical 960×720 出图）。
   → W2 的统一覆盖点必须放在 `_views_of` **之后**，且 ir 与 optical 尺寸不一致时要打印明确警告。
2. **W1 有一个必须同步改的回归地雷**：`scenegen.py:1609` 的俯视布点写死 4:3
   （`_span_y = span_m * (720.0/960.0)`）。只在工作台加宽高输入框而不改这一行，放开 16:9 后
   **点击放置的世界 Y 坐标会错**。同理 `studio.py:1451-1452` 的布点预览强制 4:3 且 ≤960。
3. **裸 API 注入尺寸会产出自相矛盾的场景**：直接发 narrow_width=1920（前端做不到，只有裸 API 能）
   得到 cams 1920×1080，但 `clip.views` 被 `studio.py:1144` 在 views_mode=single 时 pop 成 **null**。
   → W1 落地时要一并处理这个 pop，否则新入口会复现同一个不一致。
4. **SAR 欠采样的确切机制**：`sar.py:300-304` 写的是
   `az_ext = float(rc["az_extent_m"]) or (n_cross*rho_a/amp_os)`——**只有 az_extent_m 缺失时才理会
   oversample**，而两个已发布 run 都设了它。实测每分辨率单元像素数 0.75–0.80（Nyquist 需 ≥2），
   欠采样轴的斑点自相关变成白的，而 `sar.py:487-495` 自己就警告过这「看起来像传感器噪声而不是 SAR」。
   修法约 5 行：即使给了 az_extent_m 也要执行 oversample。
5. **过期尺寸说法还有两处未清**：上一轮已把 README 改到 768/640，但 `docs/导引头视角.md` 仍有 6 处
   960×720 / 1280×960，`docs/平台设计调研.md:319` 仍用 960 算 IFOV。而 960×720/1280×960 恰恰是
   **工作台真实产出的尺寸**——文档写的是工作台默认值，README 写的是策划场景值，两边都不是
   「用户能指定的值」。→ 并入 W18，新增「分辨率怎么定」一节时一起清。
6. **全仓库没有 GSD 这个概念**（`gsd|ground_sample|GSD` 在一方代码里 0 命中），且内参 K 仅在
   `research_baseline` 为真时产出 → **32 个策划场景只有 4 个有 K**（studio 生成的都有）。
   → W11/W12/W13 是一组：PNG 写 pHYs+tEXt、frames.json 补 W/H/FOV/K、GSD=2·range·tan(fov/2)/W。

**同时记下不要过度修正的部分**（终版取证 §3，八项均有实测支撑）：`thermal.py:183-292` 是真正的牛顿
热平衡解算（只是被 temp_c 覆盖）；`materials.py` 是真正的多波段库且 MWIR/LWIR 发射率已分开
（painted_metal 0.85/0.92，即第 8 条所需数据已存在）；`lidar.py` 是真链路方程激光雷达、用**伽马分布**
多视斑点（正确的相干回波统计，不是高斯）；`sar.py` 产出的是**标定过的 γ⁰ 产品**（斜距列分箱、
逐目标散射子扇按自身命中点斜距分箱、射线投影阴影、PSF 相关复斑点归一到单位均值功率、
由真实雷达方程反演 NESZ）；`camera.py` 的噪声底**测得是对的**（最平图块 std 1.19–1.33 DN、
八帧时域 std 1.03–1.09 DN，两个独立估计量一致，对上文档给的 ISO-100 目标 1.5–1.7 DN）；
`render_optical.py` 有 Nishita 物理天空、由立体角比推出（而非魔数）的 14× 曝光补偿、
逐帧牛顿求解的自动曝光；跨模态配准架构是对的（一个 world / 一个 BVH 喂五个波段，
`dataset.py:701` 逐帧调 thermal.solve()）——**这套架构不要动，要动的只是最后一米的显示与量化，
以及三处用可见光照片冒充仿真测量的地方。**
