# 素材来源

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

## 参考照片

光学亮度标定用了 11 张公开的无人机实拍照片（仅用于统计天空与目标的亮度分布，
不进入仓库、不参与再分发）。

## 谱库

红外发射率使用文献值（见 `simkit/materials.py`），可选接入 NASA ECOSTRESS
光谱库（`tools/import_speclib.py`，需自行申请下载）。
