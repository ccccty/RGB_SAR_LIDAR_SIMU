# 真实背景板（backdrops）

空中俯拍仿真目标合成用的真实照片背景。所有图均为 **CC0 / Public domain**，
作者与许可逐条记在 [catalog.json](catalog.json) 中。

## 目录

| 目录 | 内容 | 张数 |
| --- | --- | --- |
| `forest/` | 森林/林地，俯拍冠层、林间道路、林区湖面 | 5 |
| `city/` | 城市/城镇，航拍天际线与街区俯视 | 6 |
| `port/` | 港口/码头，集装箱码头、起重机、防波堤、游艇港 | 6 |
| `island/` | 岛屿/海岸线，珊瑚礁、环礁、岩岸 | 6 |
| `open/` | 开阔海面，浪纹与涌浪纹理 | 6 |

合计 29 张，约 25 MB。

## 规格

- 原始来源多为 3000～7000 px 的 Wikimedia Commons 文件；
  取 2048 px 宽渲染图，转 JPEG q92 后落盘（`catalog.json` 里的
  `width`/`height` 是 **落盘后实测** 的尺寸，均 ≥ 1024×768、横构图）。
- 每张图都用 PIL 完整解码校验过；解码失败或小于下限的文件会删除重下。
- 命名：`<两位序号>_<标题 slug>.jpg`。

## 来源与流程

素材全部来自 Wikimedia Commons，脚本复用 `tools/fetch_refs.py` 的做法
（`generator=search` + `prop=imageinfo`，取 `thumburl` 与
`extmetadata.LicenseShortName`/`Artist`）。

```bat
python tools\fetch_backdrops.py probe     :: 检索候选池，写 _candidates.json
python tools\fetch_backdrops.py stage     :: 按许可+航拍信号打分，下载前 N 张到 _stage
python tools\fetch_backdrops.py sheet     :: 拼贴图，人工过一遍
python tools\fetch_backdrops.py finalize  :: 按 picks.json 落盘并重建 catalog.json
```

搜索到的"航空照片"里有大量扫描明信片、黑白海军档案馆图、卫星云图、
甚至只是名字叫 Forest Hill 的城镇，所以流程里保留了 `stage` → 看拼贴图 →
人工挑选这一段，而不是让脚本直接抓前 5 张。

## catalog.json 字段

`file` / `category` / `title` / `author` / `license` / `license_url` /
`source_url` / `width` / `height` / `bytes` / `orig_width` / `orig_height` /
`sha256` / `aerial` / `query` / `credit` / `description` / `downloaded`
