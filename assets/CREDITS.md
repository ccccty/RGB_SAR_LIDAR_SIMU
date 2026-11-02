# 素材来源

## 2026-11-01 民航基线模型

新增 B777 - LaRC、SIERRA - ARC，来自
[NASA Airborne Science Program 三维模型库](https://airbornescience.nasa.gov/3d-models/)。
源文件/哈希见 [`aviation/sources.json`](aviation/sources.json)，尺度与转换记录见
[`aviation/geometry.json`](aviation/geometry.json)。下载和转换脚本随仓库提供，大模型不纳入 Git。

使用遵循 [NASA 媒体使用指南](https://www.nasa.gov/nasa-brand-center/images-and-media/)，
不将这批资产声明为 CC0；NASA 标识/商标和第三方权利例外仍适用，不暗示 NASA 背书。

## 3D 模型

| 文件 | 来源 | 许可 | 原始三角面 |
|---|---|---|---|
| `global_hawk.obj` | NASA 3D Resources — Global Hawk | 公有领域 | 37,120 |
| `rover.obj` | NASA 3D Resources — Mars Perseverance Rover | 公有领域 | 199,601 |
| `ingenuity.obj` | NASA 3D Resources — Mars Helicopter Ingenuity | 公有领域 | 55,000 |
| `ship_medium.obj` | Poly Haven — dutch_ship_medium | CC0 | 69,162 |
| `concrete_road_barrier.obj` | Poly Haven — concrete_road_barrier | CC0 | 60,928 |
| `modular_chainlink_fence.obj` | Poly Haven — modular_chainlink_fence | CC0 | 89,232 |
| `shrub_01.obj` | Poly Haven — shrub_01 | CC0 | 156,012 |

NASA 的 3D 资源属美国联邦政府作品，不受版权保护（公有领域）。
Poly Haven 的全部资产以 CC0 发布，可商用、可修改、无需署名。

高模在生成时用 `scene.prepare_mesh()` 自动做顶点聚类抽稀（`decimate`），
默认抽到几千面，保证轮廓与雷达/激光的面元散射仍有意义。

## 2026-11-02 载具资产库（assets/models）

按载体类别归档的 %d 个模型（旋翼/固定翼无人机、直升机、民航与军用飞机、地面
车辆、军用地面车辆、军舰、民用船舶），全部来自 [Poly Pizza](https://poly.pizza/)
聚合的公开模型，逐条清单与作者见 [`models/catalog.json`](models/catalog.json)
与 [`models/README.md`](models/README.md)。

| 许可 | 数量 | 义务 |
|---|---|---|
| CC0 1.0 | 13 | 可商用、可修改、无需署名 |
| CC-BY 4.0 | 39 | 需保留作者署名（下表） |

CC-BY 模型的作者署名：

| 作者 | 模型 |
|---|---|
| Aaron Clifford | 多旋翼无人机 |
| Alex Safayan | 游艇、集装箱船 |
| Christopher Burgess (ChrispyBurgers) | 航拍无人机 |
| J-Toastie | 军用运输卡车 |
| Joshua Johanson | 双翼飞机 |
| LEDEGENERE | 远洋邮轮 |
| NateGazzard | 四旋翼无人机 |
| Nick Olson | 小型四旋翼 |
| Nico _ | 轻型坦克 |
| Poly by Google | 厢式货车、双发战斗机、双发窄体客机、四发宽体客机、城市公交、多用途战斗机、家用轿车、帆船、常规潜艇、支线客机、攻击核潜艇、无人机机体、游轮、紧凑型轿车、货运飞机、通用直升机、高速截击机 |
| Silly Fear | 四旋翼无人机改型 |
| Thomas de Rivaz | 战列舰 |
| Zayn | 中型坦克 |
| Zsky | 导弹艇、武装直升机 |
| angelo raffaele Catalano | 战列舰改型 |
| jeremy | 单发战斗机、大型游轮、涡桨支线机、运输直升机 |
| madtrollstudio | 高机动越野车 |
| sftwr314r8 | 中空长航时无人机 |

几何经过缩放与抽稀后重新导出（OBJ，无贴图），属衍生作品；原始 GLB 不入库，
可按 `tools/fetch_vehicle_models.py` 重跑复现。

## 参考照片

光学亮度标定用了 11 张公开的无人机实拍照片（仅用于统计天空与目标的亮度分布，
不进入仓库、不参与再分发）。

## 谱库

红外发射率使用文献值（见 `simkit/materials.py`），可选接入 NASA ECOSTRESS
光谱库（`tools/import_speclib.py`，需自行申请下载）。
