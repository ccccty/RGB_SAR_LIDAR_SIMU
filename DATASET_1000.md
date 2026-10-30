# 1000 帧仿真数据集 (dataset_1000)

由 simkit 离线生成的 1000 帧多模态目标检测数据集。全部为物理仿真结果，不含任何实拍素材。

## 内容

| 场景 | 目标类别 | 帧数 | 传感器 |
|---|---|---|---|
| ds_air_uav | uav | 360 | 可见光 / MWIR / LWIR / 雷达 / LiDAR |
| ds_ground_vehicle | vehicle | 340 | 同上 |
| ds_sea_boat | boat | 300 | 同上 |
| **合计** | 3 类 | **1000** | 4 个 COCO + 雷达图 |

## 目录结构

```
dataset_1000/
  coco_optical.json      COCO：可见光 1000 图 / 1000 标注
  coco_ir_mwir.json      COCO：中波红外 3-5 µm，1000 / 1000
  coco_ir_lwir.json      COCO：长波红外 8-12 µm，1000 / 1000
  coco_lidar.json        COCO：激光雷达距离图，1000 / 938
  scenes.json            每个场景的统计与元信息
  ds_air_uav/
    meta.json  radar_frames.json
    images/optical        可见光 RGB 640x480
    images/ir_mwir        16 位红外原始计数 (另有 _preview 8 位预览)
    images/ir_lwir
    images/radar          距离-多普勒图
    images/radar_ra       距离-角度图（含 ra_peak 峰值坐标）
    lidar/range           激光雷达距离图 480x120
    lidar/intensity       激光雷达强度图
```

## 标注

标准 COCO 格式，`bbox` 为 `[x, y, w, h]`。类别 id 1/2/3 = uav/vehicle/boat。
所有标注来自仿真器的物体 ID 缓冲（逐像素掩码），不是手工框。

雷达模态不输出 COCO：`radar_frames.json` 每帧给出 3D 框、检测点云（距离/方位/俯仰/径向速度/SNR）和杂波计数。

## 验证结果

```
optical   imgs 1000 anns 1000 orphan 0 missing 0
ir_mwir   imgs 1000 anns 1000 orphan 0 missing 0
ir_lwir   imgs 1000 anns 1000 orphan 0 missing 0
lidar     imgs 1000 anns  938 orphan 0 missing 0
光学 1000 帧两两不同（MD5 全唯一）
```

## 训练

```python
from pycocotools.coco import COCO
c = COCO("dataset_1000/coco_optical.json")
```

YOLO 转换时注意 `file_name` 是相对 COCO 文件所在目录的路径（含场景子目录）。

## 物理参数

- 雷达：FMCW 实波束，X/Ku/K/Ka/W/W94 六频段可选，逐面片 PO 散射 + Kirchhoff 粗糙度 + 大气/雨衰减
- 红外：逐面稳态热平衡 + Swinbank 天空温度 + Planck 波段积分 + 分波段发射率 + NUC/FPN 固定图案噪声
- 天气：clear / cloudy / rain / fog / storm，影响能见度、太阳辐照、天空温度、雨衰
- 域随机化：目标在相机视场内随机放置（受场景遮挡检测约束）、传感器位姿抖动、逐帧温度/太阳方位/雨强变化

超过 20000 帧可按 `--part` 分块并行生成后 `simkit merge` 拼接。
