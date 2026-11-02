# assets/figures 图册

20 张示意图，供在线演示站点直接引用。全部由 **tools/make_figures.py** 生成，
图上每个数字都能回溯到仓库里的文件（见 **index.json** 的 source 字段）。

## 怎么重现

    # 全部 20 张（纯 CPU，约 15 秒）
    env\Scripts\python.exe tools\make_figures.py

    # 只重画某几张
    env\Scripts\python.exe tools\make_figures.py --only 03,05

依赖 numpy、Pillow、matplotlib==3.11.2。中文字体在脚本启动时用
font_manager.addfont() 注册（D:/devtools/fonts/NotoSansCJKsc-Regular.otf、
-Bold.otf、NotoSansMonoCJKsc-Regular.otf），不依赖系统字体缓存。
脚本只写 assets/figures/；中间调试图落在 _tmp/figs/。

## 风格约定（20 张完全一致）

| 项 | 值 |
|---|---|
| 画布 | 1600×1000 px，RGB，figsize=(16,10)、dpi=100 |
| 背景 / 坐标区 / 网格 | #0b0f14 / #0e141b / #1b2530，网格 0.6 pt 虚线 |
| 正文 / 次要文字 | #c8d9ea / #8aa2b8 |
| 强调 / 次强调 / 警示 / 错误 / 紫 | #37c2ff / #43d98a / #ffb454 / #ff5d6c / #c792ea |
| 字号 | 轴标题 15、刻度 13、图例 13、注释 12.5–14、面板小标题 17 粗体 |
| 留边 | 左右 6%、上下 8%（脚本常量 L/R/B/T） |
| 水印 | 每张左下角 simkit · 图 NN，8 pt #5b7186 |

图内不画中文大标题（站点渲染标题），但坐标轴、图例、面板标注与单位一律齐备。
右下角那行灰色小字是数据来源，不是水印。

## 每张图的数据来源

| 图 | 文件 | 数据来源 |
|---|---|---|
| 01 | fig01_modality_wall.png | data/manifest.json（seeker_air2ground 第 8 帧）+ data/seeker_air2ground/*.png |
| 02 | fig02_sar_geometry.png | data/seeker_air2ground/cfg.json、manifest sar_meta、docs/SAR成像.md 第 2.2 节 |
| 03 | fig03_sar_sequence.png | data/seeker_air2ground/sar/*.png + 各帧 boxes |
| 04 | fig04_radar_stretch.png | data/seeker_air2ground/radar_ra/000007.png + radar.peak |
| 05 | fig05_pointcloud_views.png | data/cloud/final_ground.json 第 1 帧 |
| 06 | fig06_seeker_trajectory.png | manifest 三个导引视角场景 × 20 帧 + PLAN.md P0-2 |
| 07 | fig07_radar_budget.png | data/radar_bands.json + tools/radar_budget_table.py 的雷达方程 |
| 08 | fig08_rain_attenuation.png | data/radar_bands.json 的 rain_k / rain_a（ITU-R P.838-3） |
| 09 | fig09_atmosphere_transmittance.png | simkit/atmosphere.py + simkit/weather.py 的 derive() |
| 10 | fig10_label_consistency.png | datasets/yolo/export.json 的 label_consistency |
| 11 | fig11_dataset_composition.png | datasets/yolo/export.json |
| 12 | fig12_box_size.png | datasets/yolo/labels/optical 与 optical_wide（train+val） |
| 13 | fig13_mosaic.png | datasets/yolo/images/optical/train + 对应 labels |
| 14 | fig14_composite.png | datasets/yolo/composites/stack_narrow/train + stacks |
| 15 | fig15_coverage.png | data/manifest.json 的 mods / boxes 统计 |
| 16 | fig16_pipeline.png | README.md、docs/连贯数据集与工作台.md、simkit 模块划分 |
| 17 | fig17_sensor_synergy.png | fig07 的雷达模型 + fig09 的消光系数 |
| 18 | fig18_target_crops.png | data/seeker_air2ground 第 8 帧六模态 PNG |
| 19 | fig19_target_px_range.png | manifest 的 range_m 与 views.*.target_px |
| 20 | fig20_scale.png | manifest + export.json + simkit/selftest.py |

## 显示约定（不是数据加工）

雷达、SAR、红外、LiDAR 的 PNG 动态范围低或点稀疏，直接线性显示会全黑。
图上只做**百分位拉伸**（图注写明 P1–P99.9 之类的口径），LiDAR 稀疏点另加
一次 3×3 灰度膨胀；不做插值放大、不做平滑、不改任何数据文件。
原始文件路径见每张图右下角。

## 与最初计划的差异

- **fig06**：原计划 4 个导引视角场景，仓库当前只有 3 个（城市遮挡场景尚未重生成），
  图上按 3 个场景绘制并在副标题注明。
- **fig05**：点云取自 data/cloud/*.json（manifest 的 cloud 字段已外置），
  数组单位仍是 cm，脚本换算为 m。
