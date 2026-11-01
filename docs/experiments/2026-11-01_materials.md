# 2026-11-01 材质与模型接入真实性改进

本记录为民航机载数据生成升级的材质子任务。未提交、未推送；没有改写旧生成结果。

## 问题与修改

此前 OBJ 导入只读取顶点和三角面，`usemtl`、分组、UV 与 MTL 被丢弃。
模型在 `prepare_mesh()` 缩放/简化/归一化后进一步只剩几何，Blender 全物体使用一个预设颜色。
NASA GLB 本身带有基础色、金属度/粗糙度、法线等纹理，但既有 OBJ 转换输出的 MTL 不含这些贴图。

此次修改：

- `simkit/mesh.py` 保留三角面对应的源材质名、分组名、每角 UV、MTL 外观属性。
  多边形三角化和负索引导入保持相同属性关联；无效零索引显式报错。
- `Mesh.with_geometry()` 贯穿旋转、平移、缩放、高度归一化、网格简化。
  简化删面时用同一个布尔掩膜选择 UV/源材质/分组，避免标签错配。
- `prepare_mesh()` 统一删除重复/退化三角面，确保 Blender `Mesh.validate()` 不再单独改面序。
  初轮真实主流水线发现 B777 简化网格有 616 个重复面、SIERRA 有 317 个；此前的材质
  单元测试未包含 Blender validate。现在已补上实际 validate 后面序/数量完全相同的检查。
  这同时避免物理散射模型把重叠重复三角面累加为额外回波。
- `simkit/materials.py` 提供显式 `material_map`，将源材质名映射到已有物理材质库。
  不从可见光颜色猜测发射率或雷达反射率；未映射源名使用对象声明的 `mat`，并记录未映射名。
- 增加 `plastic` 工程假设预设，用于明确的干燥、不填充、不导电、不透明聚合物先验。
  RF 幅度为假设相对介电常数 2.5 的法向 Fresnel 反射幅度约 0.225；它不是 RCS。
  发射率、吸收率、颜色等均明确标为可配置研究假设，非实测谱参数，不适用于所有塑料。
  使用该材质时，`source_material_report.material_assumptions` 会保存这一限制。
- `simkit/scene.py` 为 World 提供 `face_source_material`、`face_group`、
  `face_local_centers`，保留对象 `thermal_regions`，支持红外模块按真正部件配置热源。
- `blender/material_helpers.py` 为同一份 prepared mesh 设置逐面材质和 UV。
  可以从 MTL 读取基础色、粗糙度、金属度和常见贴图，也可从原 GLB 复用完整 PBR 节点。
  GLB 只用于提供材质，导入对象立即删除，几何仍由共用的 prepared mesh 提供。
  缺失源材质、贴图或 UV 被记录，纹理面没有有效 UV 时使用明确的对象预设回退。

旧配置默认不启用源外观，维持原单材质行为；新配置需要显式启用。

## 配置接口

```json
{
  "shape": "obj",
  "model": "../assets/aviation/airliner.obj",
  "mat": "painted_metal",
  "import_materials": true,
  "optical_material_source": "../assets/aviation/airliner.glb",
  "material_map": {
    "PaintM": "painted_metal",
    "PaintPM": "painted_metal",
    "MetalM": "metal_bare",
    "MetalPM": "metal_bare",
    "GlassM": "glass",
    "GlassPM": "glass"
  },
  "thermal_regions": [
    {"group_names": ["engine"], "heat_flux_w_m2": 900}
  ]
}
```

这只是接口示例，不代表 NASA 模型拥有名为 `engine` 的分组或此热通量经标定。
实际源材质、分组和匹配面数必须检查。塑料等未在物理库中精确建模的材料不应随意映射。
红外 `thermal_regions` 的匹配和热学行为由红外模块实现。

Blender 主脚本接线：

```python
from material_helpers import apply_source_materials
report = apply_source_materials(ob, prepared_mesh, spec, cfg["_base_dir"])
```

应在 `add_mesh_object()` 后立即调用，保持面序与 prepared mesh 一致。

## 验证

在项目根目录执行：

```powershell
.\env\Scripts\python.exe -m unittest discover -s tests -p test_materials.py -v
.\third_party\blender\blender.exe -b --factory-startup -P tests/blender_material_probe.py
.\env\Scripts\python.exe -m simkit selftest
```

结果：

- 八项材质单元测试全部通过：三角化/负索引 UV、变换/简化属性关联、物理映射与回退、
  无效映射和无效索引拒绝、旧单材质兼容、塑料工程假设、重复/退化面清理。
- Blender 4.5.9 LTS 对 NASA 两个真实下载资产完成接入检查：
  初轮 B777 简化后 4314 面、23 个材质纹理节点；SIERRA 4872 面、36 个材质纹理节点。
  加入共享重复面清理后分别为 3698 面/19 节点、4555 面/36 节点。
  实际执行 Blender `data.validate()` 后面序和计数保持不变。
  所有保留面具有源材质和有效 UV，图像像素可读取，源 GLB 对象无泄漏。
- 原有 `simkit selftest` 全部通过。
- Blender 接入检查结果写入 `out/_tmp/materials/probe_results.json`（不进 Git）。

## 适用范围与限制

这些检查证明材质/UV管线没有丢失对应关系，不等于已经证明渲染与真实飞机一致。
顶点聚类简化会改变几何与纹理插值，UV 按原面角保留属于近似；正式数据仍需检查轮廓和纹理失真。
当前 OBJ 导入未使用原始自定义分裂法线；MTL 的透明度仅保留为元信息，不自行推断玻璃透射。
原 GLB 节点方式可保留源透明/清漆等行为，但其物理正确性仍取决于源资产。

光学 appearance 与物理 material_map 是两条明确配置的通道：恢复原 PBR 纹理不意味着
自动获得真实发射率、雷达 RCS 或激光反射率。

主控热区初稿的几何检查（旧朝向、共享清理前）：B777 左/右区域分别匹配 690/710 面，
面积约 455/445 m²；SIERRA 区域匹配 1238 面、约 2.33 m²，且包含玻璃/橡胶等多种材质。
因此这些宽泛局部包围盒只能标作 research 热源区域，不能宣称已经定位真实发动机。
模型朝向纠正后需要重新定位热区并检查有效匹配数量；零匹配区域不得静默接受。
