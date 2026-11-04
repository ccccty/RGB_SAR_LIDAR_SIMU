/* simkit studio — 收纳箱工作台
   纯 ES5 · 零依赖 · 用 XMLHttpRequest 而不是 fetch，老浏览器也能跑。

   状态只有一份：S。所有控件写 S，摘要、预检、提交都从 S 生成，
   不反读 DOM，所以“收起时看到的取值”和“真正提交的东西”永远一致。 */
(function () {
  "use strict";

  var API = "/api";
  var BG_HDRI_ID = "kloofendal_48d_partly_cloudy_puresky";
  var SCALE = { small: 0.7, medium: 1.0, large: 1.4 };
  var SIZE_LABEL = { small: "小", medium: "标准", large: "大" };
  var COMMON = ["id", "pos", "vel", "accel", "life", "band", "visible", "note"];
  var MODE_LABEL = { none: "无干扰", active: "有源", passive: "无源", mixed: "混合" };
  var AUGMENT_LABEL = {
    inherit: "继承场景", off: "关闭", active: "有源类", passive: "无源类", mixed: "混合"
  };
  var AUGMENT_HINT = {
    inherit: "沿用场景文件自带的干扰增广设置。",
    off: "不做任何图像级干扰增广。",
    active: "叠加有源类增广（噪声压制、扫描线等）。",
    passive: "叠加无源类增广（眩光、散斑、遮挡等）。",
    mixed: "有源与无源类增广混合叠加。"
  };
  var JOB_LABEL = { queued: "排队中", running: "渲染中", done: "已完成",
                    failed: "失败", cancelled: "已取消" };

  var DEFAULT_OPT = {
    weather: [
      { key: "clear", label: "晴" }, { key: "cloudy", label: "多云" },
      { key: "rain", label: "雨" }, { key: "fog", label: "雾" },
      { key: "mixed", label: "混合" }, { key: "evolve", label: "演变" }
    ],
    sensors: [
      { key: "optical", label: "可见光", note: "Blender 渲染，最慢" },
      { key: "ir", label: "红外", note: "中波 + 长波，热辐射" },
      { key: "radar", label: "雷达", note: "距离-角度 / 距离-多普勒" },
      { key: "lidar", label: "激光雷达", note: "距离图 + 强度 + 点云" },
      { key: "sar", label: "SAR", note: "合成孔径，需后端集成" }
    ],
    env_kinds: [
      { key: "sky", label: "天空" }, { key: "ground", label: "地面" },
      { key: "sea", label: "海洋" }
    ],
    aspects: [
      { key: "head_on", label: "迎头拍摄" }, { key: "pursuit", label: "追击拍摄" }
    ],
    views: {
      narrow: { fov_deg: 2.2, width: 960, height: 720 },
      wide: { fov_deg: 18.0, width: 1280, height: 960 }
    },
    ecm: { kinds: [], groups: [{ key: "active", label: "有源" },
                               { key: "passive", label: "无源" }] },
    cost: { sec: { optical: 9, ir: 2.5, radar: 0.8, lidar: 0.6, sar: 1.6 },
            mb: { optical: 2.4, ir: 1.3, radar: 0.15, lidar: 0.05, sar: 1.5 } },
    limits: {},
    default_out: "out/studio_runs",
    project_root: ""
  };

  /* ECM 参数字典：后端键名 -> 中文。未收录的键会退化成原键名。 */
  var PARAM_LABEL = {
    erp_w: "等效辐射功率 ERP（W）",
    jam_bandwidth_mhz: "干扰带宽（MHz）",
    mode: "压制方式",
    look_at_sensor: "对准传感器",
    sidelobe_db: "旁瓣电平（dB）",
    off_band_factor: "带外因子",
    rcs_m2: "雷达截面积（m²）",
    count: "数量",
    power_ratio: "功率比",
    range_offset_m: "距离偏移（m）",
    range_step_m: "距离步进（m）",
    velocity_offset_mps: "速度偏移（m/s）",
    velocity_step_mps: "速度步进（m/s）",
    angle_offset_deg: "角度偏移（°）",
    rate_mps: "拖引速率（m/s）",
    max_offset_m: "最大距离偏移（m）",
    max_offset_mps: "最大速度偏移（m/s）",
    radius_m: "半径（m）",
    growth_mps: "膨胀速率（m/s）",
    decay_s: "衰减时间（s）",
    dilution_exponent: "稀释指数",
    fall_mps: "下落速度（m/s）",
    size_m: "尺寸（m）",
    temp_c: "温度（°C）",
    tether_m: "系绳长度（m）",
    towed_by: "拖曳目标",
    size: "尺寸 XYZ（m）",
    mat: "材质",
    anchor: "锚点",
    wind: "风 XYZ（m/s）",
    aspect: "椭球比 XYZ"
  };

  var OPT = DEFAULT_OPT;
  var ASSETS = [];
  var BACKDROPS = [];
  var ASSET_ERROR = "";
  var RUNS = [];
  var RECIPES = [];

  var PF = null;          // 最近一次 /api/preflight 的响应
  var PF_LOCAL = [];      // 前端先行发现的错误
  var PF_STATE = "idle";  // idle | checking | ok | warn | error | offline
  var PF_SEQ = 0;
  var pfTimer = null;

  /* 输出目录浏览器的即时状态，与预检结论分开：浏览管“能不能写/是否存在”，
     预检管“会不会被拒绝/会不会改名”。 */
  var OUT = { info: null, checking: false, error: "", localBad: false,
              localText: "", open: false, seq: 0 };
  var browseTimer = null;

  var SUBMITTING = false;
  var currentJob = null;
  var activeJobId = "";
  var activeKind = "";
  var pollTimer = null;
  var pollFails = 0;

  var planData = null;
  var planBusy = false;
  var toastTimer = null;

  var S = defaultState();

  function defaultState() {
    return {
      name: "我的数据集",
      env: { kind: "sky", weather: "clear" },
      /* file 为空时按种子在同类照片间轮换；选定具体照片则永远用它。 */
      bg: { mode: "none", path: "", dataUrl: "", category: "forest", file: "" },
      bgUi: { open: "", domains: {} },
      /* 在背景图上的放置：uv 是归一化图面坐标，orig 为左上角。
         points 是同一条轨迹的 uv 序列。 */
      /* 自定义模式下每个目标各自一份：在背景图上的位置 + 一条轨迹。
         targets 的长度始终等于 obj.count。 */
      targets: [],
      targetIdx: 0,
      /* placed 只有在用户真的在图面上点过之后才为 true：否则默认的 0.5/0.5
         会让"只要选了背景库就强制正俯视"，把迎头/追击都盖掉。 */
      place: { mode: "place", u: 0.5, v: 0.5, points: [], busy: false, url: "", placed: false,
               speed: 80 },
      obj: {
        category: "军用飞机", asset: "军用飞机/双发战斗机.obj",
        cls: "fighter", scale: 1.0, count: 1
      },
      size: "medium",
      ecm: { entities: [] },
      /* 干扰增广是信号级增强（噪声/眩光/遮挡），与上面的物理干扰实体是两件事：
         实体会进 3D 场景，增广只改图像。后端 _parse 读顶层 interference_mode。 */
      interferenceMode: "inherit",
      traj: {
        mode: "preset", points: [], smoothed: null,
        alt_m: 2500, speed_mps: 250, closed: false,
        stats: null, warnings: [], error: "", optimizing: false
      },
      view: { range_km: 8.0, aspect: "head_on", fov_deg: 2.2, views: "single" },
      capture: {
        sensors: ["optical", "ir", "radar"], frames: 20, fps: 10.0, seed: 7
      },
      output: { preview: true, coco: true, yolo: true, readme: true },
      outputDir: "",
      recipeSource: ""
    };
  }

  /* ------------------------------------------------------------ 基础工具 */

  function $(id) { return document.getElementById(id); }
  function ce(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }
  function clear(node) {
    if (!node) { return; }
    while (node.firstChild) { node.removeChild(node.firstChild); }
  }
  function each(list, fn) {
    for (var i = 0; i < list.length; i++) { fn(list[i], i); }
  }
  function setText(id, text) {
    var node = $(id);
    if (node) { node.textContent = text; }
  }
  function show(node, yes) { if (node) { node.hidden = !yes; } }
  function has(obj, key) { return Object.prototype.hasOwnProperty.call(obj, key); }
  function isArray(v) { return Object.prototype.toString.call(v) === "[object Array]"; }
  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
  function num(v, def) {
    var x = parseFloat(v);
    return isFinite(x) ? x : def;
  }
  function intval(v, def) {
    var x = parseInt(v, 10);
    return isFinite(x) ? x : def;
  }
  function copy(obj) { return JSON.parse(JSON.stringify(obj)); }
  function indexOf(list, value) {
    for (var i = 0; i < list.length; i++) {
      if (list[i] === value) { return i; }
    }
    return -1;
  }
  function trim(v) { return String(v === null || v === undefined ? "" : v).replace(/^\s+|\s+$/g, ""); }

  function labelIn(list, key, fallback) {
    for (var i = 0; i < (list || []).length; i++) {
      if (String(list[i].key) === String(key)) {
        return list[i].label || String(key);
      }
    }
    return fallback !== undefined ? fallback : String(key);
  }

  function conn(ok) {
    var box = $("conn");
    var text = $("connText");
    if (!box || !text) { return; }
    if (ok) {
      box.className = "conn online";
      text.textContent = "本地服务已连接";
    } else {
      box.className = "conn offline";
      text.textContent = "后端未连接";
    }
  }

  function toast(message) {
    var box = $("toast");
    if (!box) { return; }
    box.textContent = String(message);
    box.className = "toast show";
    if (toastTimer) { clearTimeout(toastTimer); }
    toastTimer = setTimeout(function () { box.className = "toast"; }, 3400);
  }

  /* 所有请求走这里：4xx/5xx 也返回 {error}，网络失败标记 offline。 */
  function ajax(method, path, body, cb) {
    var xhr = new XMLHttpRequest();
    var fired = false;
    function finish(err, data) {
      if (fired) { return; }
      fired = true;
      cb(err, data);
    }
    xhr.open(method, path, true);
    xhr.setRequestHeader("Accept", "application/json");
    if (body !== null && body !== undefined) {
      xhr.setRequestHeader("Content-Type", "application/json");
    }
    xhr.timeout = 25000;
    xhr.onreadystatechange = function () {
      if (xhr.readyState !== 4) { return; }
      var data = null;
      var text = xhr.responseText || "";
      try {
        data = text ? JSON.parse(text) : {};
      } catch (e) {
        data = { error: text ? text.slice(0, 400) : ("HTTP " + xhr.status) };
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        conn(true);
        finish(null, data || {});
        return;
      }
      if (xhr.status > 0) { conn(true); }
      if (!data || typeof data !== "object") { data = {}; }
      if (!data.error) { data.error = "HTTP " + xhr.status; }
      finish(data);
    };
    xhr.onerror = function () { conn(false); finish({ error: "无法连接后端", offline: true }); };
    xhr.ontimeout = function () { finish({ error: "请求超时", offline: true }); };
    xhr.send(body !== null && body !== undefined ? JSON.stringify(body) : null);
  }

  /* ------------------------------------------------------------ 选项查找 */

  function envLabel(key) { return labelIn(OPT.env_kinds, key, key); }
  function weatherLabel(key) { return labelIn(OPT.weather, key, key); }
  function sensorLabel(key) { return labelIn(OPT.sensors, key, key); }
  /* 当前目标是否站在地面/海面上：决定哪些视角可用。 */
  function isSurfaceTarget() {
    var cls = trim(S.obj.cls || "");
    var list = (OPT && OPT.surface_cls) || [];
    if (list.length) { return indexOf(list, cls) >= 0; }
    return cls === "vehicle" || cls === "tank" || cls === "ship";
  }

  function aspectLabel(key) {
    if (key === "look_down" || key === "top_down") { return "俯视"; }
    if (key === "head_on") { return "迎头"; }
    if (key === "pursuit") { return "追击"; }
    return labelIn(OPT.aspects, key, key);
  }
  function sizeForScale(scale) {
    if (Math.abs(scale - SCALE.small) < 1e-6) { return "small"; }
    if (Math.abs(scale - SCALE.medium) < 1e-6) { return "medium"; }
    if (Math.abs(scale - SCALE.large) < 1e-6) { return "large"; }
    return "custom";
  }
  function findAsset(path) {
    for (var i = 0; i < ASSETS.length; i++) {
      if (ASSETS[i].path === path) { return ASSETS[i]; }
    }
    return null;
  }
  function assetName(path) {
    var a = findAsset(path);
    if (a) { return a.name; }
    var parts = String(path || "").split("/");
    var last = parts[parts.length - 1] || "";
    last = last.replace(/\.obj$/i, "");
    return last || "未选择模型";
  }
  function ecmKind(key) {
    var kinds = (OPT.ecm && OPT.ecm.kinds) || [];
    for (var i = 0; i < kinds.length; i++) {
      if (kinds[i].key === key) { return kinds[i]; }
    }
    return null;
  }
  function ecmKindLabel(key) {
    var k = ecmKind(key);
    return k ? k.label : String(key || "");
  }
  function ecmDefaults(key) {
    var k = ecmKind(key);
    return k && k.defaults ? k.defaults : {};
  }
  function ecmGroupLabel(key) {
    var groups = (OPT.ecm && OPT.ecm.groups) || [];
    return labelIn(groups, key, key === "active" ? "有源" : (key === "passive" ? "无源" : key));
  }
  function paramLabel(key) { return PARAM_LABEL[key] || String(key); }

  /* ------------------------------------------------------------ 摘要 */

  function fmtKm(v) {
    var n = Math.round(num(v, 0) * 10) / 10;
    return (Math.abs(n - Math.round(n)) < 1e-9) ? String(Math.round(n)) : n.toFixed(1);
  }
  function bgSummary() {
    if (S.bg.mode === "hdri") { return "内置天空"; }
    if (S.bg.mode === "upload") { return "自定义图"; }
    return "无背景";
  }
  function outParts() {
    var parts = [];
    if (S.output.preview) { parts.push("拼接图"); }
    if (S.output.coco) { parts.push("COCO"); }
    if (S.output.yolo) { parts.push("YOLO"); }
    if (S.output.readme) { parts.push("文档"); }
    return parts;
  }
  function ecmMode() {
    if (!S.ecm.entities.length) { return "none"; }
    var hasActive = false;
    var hasPassive = false;
    each(S.ecm.entities, function (entity) {
      var kind = ecmKind(entity.kind);
      if (kind && kind.group === "passive") { hasPassive = true; }
      else { hasActive = true; }
    });
    if (hasActive && hasPassive) { return "mixed"; }
    return hasActive ? "active" : "passive";
  }

  function ecmSummary() {
    /* 增广模式也进摘要：不给提示的话，选了"混合"而实体为空时用户会以为没生效。 */
    var aug = AUGMENT_LABEL[S.interferenceMode || "inherit"];
    var augNote = (S.interferenceMode && S.interferenceMode !== "inherit")
      ? "增广：" + aug : "";
    if (!S.ecm.entities.length) { return augNote || "无干扰"; }
    var parts = [];
    each(S.ecm.entities, function (entity) {
      var kind = ecmKind(entity.kind);
      var group = kind ? kind.group : "active";
      parts.push(ecmGroupLabel(group) + " · " + ecmKindLabel(entity.kind));
    });
    var text = parts.join(" + ");
    if (S.ecm.entities.length > 1) {
      text += "（" + S.ecm.entities.length + " 个实体）";
    }
    return augNote ? text + " · " + augNote : text;
  }

  function trajSummary() {
    if (S.traj.mode !== "drawn") {
      var fixed = fmtKm(S.view.range_km) + " km · " + aspectLabel(S.view.aspect);
      if (S.view.views === "dual") { fixed += " · 双视场"; }
      return fixed;
    }
    var text;
    if (S.traj.stats && S.traj.stats.length_m) {
      text = "手绘 · " + (num(S.traj.stats.length_m, 0) / 1000).toFixed(1) + " km";
    } else if (S.traj.smoothed) {
      text = "手绘 · " + (polylineLengthM(S.traj.smoothed) / 1000).toFixed(1) + " km";
    } else {
      text = "手绘 · " + S.traj.points.length + " 点（未优化）";
    }
    if (S.view.views === "dual") { text += " · 双视场"; }
    return text;
  }

  function renderSummaries() {
    setText("sum-scene", envLabel(S.env.kind) + " · " + weatherLabel(S.env.weather) +
      " · " + bgSummary());

    var target = assetName(S.obj.asset) + " ×" + S.obj.count;
    if (Math.abs(S.obj.scale - 1) > 1e-9) {
      target += " · " + (Math.round(S.obj.scale * 100) / 100) + "×";
    }
    setText("sum-target", target);

    setText("sum-ecm", ecmSummary());
    setText("sum-view", trajSummary());

    var sensors = [];
    each(S.capture.sensors, function (k) { sensors.push(sensorLabel(k)); });
    setText("sum-capture", (sensors.length ? sensors.join("+") : "未选模态") +
      " · " + S.capture.frames + " 帧");

    var parts = outParts();
    setText("sum-output", parts.length ? parts.join(" + ") : "无产物");
  }

  /* ------------------------------------------------------------ 配方与提交体 */

  function asciiName(raw) {
    var text = trim(raw);
    if (/^[A-Za-z0-9_.-]{1,64}$/.test(text)) { return text; }
    var slug = text.replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/^[._-]+|[._-]+$/g, "");
    slug = slug.slice(0, 64);
    return slug || "dataset";
  }

  function recipeNameValue() {
    var raw = trim(S.name);
    if (/^[A-Za-z0-9_.\-\u4e00-\u9fff]{1,64}$/.test(raw)) { return raw; }
    var cleaned = raw.replace(/[^A-Za-z0-9_.\-\u4e00-\u9fff]+/g, "_")
      .replace(/^[._-]+|[._-]+$/g, "").slice(0, 64);
    return cleaned || "dataset";
  }

  function trajectoryPayload() {
    if (S.traj.mode !== "drawn") { return { mode: "preset" }; }
    return {
      mode: "drawn",
      points: S.traj.smoothed ? copy(S.traj.smoothed) : [],
      speed_mps: num(S.traj.speed_mps, 250),
      alt_m: num(S.traj.alt_m, 2500),
      closed: !!S.traj.closed
    };
  }

  function backgroundPayload() {
    if (S.bg.mode === "hdri") {
      return { mode: "hdri", id: BG_HDRI_ID };
    }
    if (S.bg.mode === "library") {
      /* Real plate from assets/backdrops. For ground/sea the engine also maps it
         onto the ground material -- a look-down view shows terrain, not sky.
         A chosen file wins over the seed rotation the engine would otherwise do. */
      var _bg = { mode: "library", category: S.bg.category || "forest" };
      if (S.bg.file) { _bg.file = S.bg.file; }
      return _bg;
    }
    if (S.bg.mode === "upload") {
      return { mode: "upload", path: trim(S.bg.path) };
    }
    return { mode: "none" };
  }

  function entityPayload(entity) {
    var out = { kind: entity.kind };
    var key;
    for (key in entity.extra) {
      if (has(entity.extra, key)) { out[key] = entity.extra[key]; }
    }
    /* 每个实体带上该类型的完整参数，而不是只带改动项：这样导出的 recipe
       单独拿出来也能复原界面里看到的那组参数。 */
    var defaults = ecmDefaults(entity.kind);
    for (key in defaults) {
      if (has(defaults, key) && defaults[key] !== undefined) {
        out[key] = defaults[key];
      }
    }
    for (key in entity.params) {
      if (has(entity.params, key)) { out[key] = entity.params[key]; }
    }
    return out;
  }

  function buildRecipe() {
    var objects = [{
      asset: trim(S.obj.asset),
      cls: trim(S.obj.cls) || "fighter",
      scale: num(S.obj.scale, 1),
      count: clamp(intval(S.obj.count, 1), 1, 64)
    }];
    var entities = [];
    each(S.ecm.entities, function (entity) {
      entities.push(entityPayload(entity));
    });
    return {
      name: S.recipeSource || recipeNameValue(),
      environment: {
        kind: S.env.kind,
        weather: S.env.weather,
        background: backgroundPayload()
      },
      objects: objects,
      ecm: { mode: ecmMode(), entities: entities },
      view: {
        range_km: num(S.view.range_km, 8),
        aspect: S.view.aspect,
        fov_deg: num(S.view.fov_deg, 2.2),
        views: S.view.views,
        trajectory: trajectoryPayload()
      },
      capture: {
        sensors: S.capture.sensors.slice(0),
        frames: clamp(intval(S.capture.frames, 20), 1, 20000),
        fps: num(S.capture.fps, 10),
        seed: clamp(intval(S.capture.seed, 7), 0, 4294967295)
      },
      output: {
        preview: !!S.output.preview,
        coco: !!S.output.coco,
        yolo: !!S.output.yolo,
        readme: !!S.output.readme
      }
    };
  }

  function buildPayload() {
    var recipe = buildRecipe();
    /* 在背景图上放置过目标时，视角必须锁成俯视正下方：uv 只有在 top_down
       的板面坐标系里才等于像素位置。 */
    if (isCustomMode()) {
      /* 自定义模式：每个目标一条 objects 记录，各自的位置和轨迹。
         count 固定 1，目标数量由数组长度表达。 */
      recipe.view.aspect = "top_down";
      recipe.view.ground_span_m = 600;
      recipe.view.trajectory = null;
      var _sy = 600 * 0.75;
      var _objs = [];
      var _spd = num(S.place.speed, 80);
      each(S.targets, function (t) {
        if (!t.uv) { return; }
        var _o = {
          asset: t.asset || trim(S.obj.asset),
          cls: t.cls || trim(S.obj.cls) || "fighter",
          scale: num(S.obj.scale, 1),
          count: 1,
          uv: [t.uv[0], t.uv[1]]
        };
        if (t.points && t.points.length > 1) {
          /* 图上的点转成相对该目标起点的世界偏移，引擎再加上 pos。 */
          var ox = (t.uv[0] - 0.5) * 600;
          var oy = (0.5 - t.uv[1]) * _sy;
          var _pp = [];
          each(t.points, function (p) {
            _pp.push([(p[0] - 0.5) * 600 - ox, (0.5 - p[1]) * _sy - oy, 0]);
          });
          _o.trajectory = { mode: "waypoints", points: _pp,
                            speed_mps: _spd, loop: false };
        }
        _objs.push(_o);
      });
      if (_objs.length) {
        recipe.objects = _objs;
      }
    }
    var payload = {
      name: S.name,
      recipe: recipe,
      aspect: recipe.view.aspect,
      frames: recipe.capture.frames,
      sensors: recipe.capture.sensors.slice(0),
      views_mode: recipe.view.views,
      output_dir: trim(S.outputDir)
    };
    /* studio._parse 用顶层 sar_enabled 决定是否保留 sar；不传就会被静默移除。 */
    if (indexOf(recipe.capture.sensors, "sar") >= 0) {
      payload.sar_enabled = true;
    }
    /* 性能检测卡算出的并行度。后端 start_job 用它决定单进程还是分帧并行；
       不传就永远 1 路，检测结果等于白算。 */
    var W = parseInt(window.BENCH_WORKERS, 10);
    payload.workers = (isFinite(W) && W > 0) ? Math.min(Math.max(W, 1), 16) : 1;
    /* 后端 _parse 只认顶层 interference_mode；放进 recipe 会被忽略，
       工作台选的干扰增广就成了静默空转。 */
    payload.interference_mode = S.interferenceMode || "inherit";
    return payload;
  }

  function localErrors() {
    var errors = [];
    if (!trim(S.obj.asset)) { errors.push("还没有选择目标模型：在「目标」箱里选一个，或在高级里填模型路径。"); }
    if (!S.capture.sensors.length) { errors.push("至少选一个模态：在「采集」箱里打开一个传感器开关。"); }
    if (S.bg.mode === "upload" && !trim(S.bg.path)) {
      errors.push("背景图选了「本地图片」：请填写服务器能读到的绝对路径（浏览器选的文件不会上传）。");
    }
    if (S.traj.mode === "preset" && !(num(S.view.range_km, 8) > 0)) {
      errors.push("拍摄距离必须大于 0 km。");
    }
    if (S.traj.mode === "drawn") {
      if (S.traj.points.length < 2) {
        errors.push("手绘轨迹至少需要两个点：在画布上左键加点。");
      } else if (!S.traj.smoothed) {
        errors.push("手绘轨迹还没有优化：点「优化轨迹」后再生成。");
      }
    }
    if (trim(S.outputDir) && OUT.localBad) {
      errors.push(OUT.localText || "输出目录不可写。");
    }
    return errors;
  }

  function updateNameHint() {
    var hint = $("nameHint");
    if (!hint) { return; }
    var raw = trim(S.name);
    var name = S.recipeSource || recipeNameValue();
    hint.textContent = "配方名 " + name + (raw === name || S.recipeSource ? "" : "（已规范化）");
    hint.title = S.recipeSource
      ? "沿用载入配方的内部名；改数据集名会换成新的配方名。"
      : "配方名允许中文；非法字符会替换成下划线。";
  }

  function defaultOutLabel() {
    var path = OPT.default_out || "out/studio_runs";
    var root = OPT.project_root || "";
    if (root && path.indexOf(root) === 0) {
      var rel = path.slice(root.length);
      while (rel.charAt(0) === "/" || rel.charCodeAt(0) === 92) {
        rel = rel.slice(1);
      }
      path = rel;
    }
    return path.split(String.fromCharCode(92)).join("/");
  }


  /* ------------------------------------------------------------ 数据加载 */

  function loadPresets() {
    ajax("GET", API + "/presets", null, function (err, data) {
      if (err) {
        conn(false);
        OPT = DEFAULT_OPT;
        PF_STATE = "offline";
        renderAll();
        renderPreflight();
        renderEstimate(localEstimate(), true);
        updateGenEnabled();
        return;
      }
      OPT = data && typeof data === "object" ? data : DEFAULT_OPT;
      if (!OPT.ecm || !OPT.ecm.kinds) { OPT.ecm = DEFAULT_OPT.ecm; }
      if (!OPT.weather) { OPT.weather = DEFAULT_OPT.weather; }
      if (!OPT.sensors) { OPT.sensors = DEFAULT_OPT.sensors; }
      if (!OPT.env_kinds) { OPT.env_kinds = DEFAULT_OPT.env_kinds; }
      if (!OPT.aspects) { OPT.aspects = DEFAULT_OPT.aspects; }
      if (!OPT.cost) { OPT.cost = DEFAULT_OPT.cost; }
      if (!OPT.views) { OPT.views = DEFAULT_OPT.views; }
      if (!OPT.default_out) { OPT.default_out = DEFAULT_OPT.default_out; }
      if (!OPT.project_root) { OPT.project_root = DEFAULT_OPT.project_root; }
      RECIPES = OPT.recipes || [];

      /* 状态里若留着已不存在的选项，回落到第一项。 */
      if (!labelIn(OPT.env_kinds, S.env.kind, "")) { S.env.kind = OPT.env_kinds[0].key; }
      if (!labelIn(OPT.weather, S.env.weather, "")) { S.env.weather = OPT.weather[0].key; }
      var knownSensors = [];
      each(OPT.sensors, function (item) { knownSensors.push(item.key); });
      S.capture.sensors = S.capture.sensors.filter(function (k) {
        return indexOf(knownSensors, k) >= 0;
      });

      renderAll();
      schedulePreflight(0);
    });
  }

  function loadBackdrops() {
    ajax("GET", API + "/backdrops", null, function (err, data) {
      BACKDROPS = (err || !data) ? [] : (data.categories || []);
      renderScene();
    });
  }

  function loadAssets() {
    ajax("GET", API + "/assets", null, function (err, data) {
      if (err) {
        ASSETS = [];
        ASSET_ERROR = err.error || "资产库暂不可用";
        renderTarget();
        return;
      }
      ASSETS = data.assets || [];
      ASSET_ERROR = data.error || "";
      if (ASSETS.length) {
        var cur = findAsset(S.obj.asset);
        if (!cur) {
          var preferred = null;
          for (var i = 0; i < ASSETS.length; i++) {
            if (ASSETS[i].path === "军用飞机/双发战斗机.obj") { preferred = ASSETS[i]; break; }
          }
          if (!preferred) {
            for (var j = 0; j < ASSETS.length; j++) {
              if (ASSETS[j].category === "军用飞机") { preferred = ASSETS[j]; break; }
            }
          }
          if (!preferred) { preferred = ASSETS[0]; }
          S.obj.asset = preferred.path;
          S.obj.category = preferred.category;
          S.obj.cls = preferred.cls;
        }
      }
      renderTarget();
      updateNameHint();
    });
  }

  /* ------------------------------------------------------------ 渲染：场景 */

  function renderScene() {
    var envSeg = $("envSeg");
    clear(envSeg);
    each(OPT.env_kinds || [], function (item) {
      var btn = ce("button", S.env.kind === item.key ? "on" : "", item.label);
      btn.type = "button";
      btn.onclick = function () {
        S.env.kind = item.key;
        renderScene();
        touch();
      };
      envSeg.appendChild(btn);
    });

    var weather = $("weatherChips");
    clear(weather);
    each(OPT.weather || [], function (item) {
      var btn = ce("button", "chip" + (S.env.weather === item.key ? " on" : ""), item.label);
      btn.type = "button";
      btn.onclick = function () {
        S.env.weather = item.key;
        renderScene();
        touch();
      };
      weather.appendChild(btn);
    });

    var modes = [
      { key: "none", label: "无背景" },
      { key: "hdri", label: "内置天空" },
      { key: "library", label: "背景库" },
      { key: "upload", label: "本地图片" }
    ];
    var bgSeg = $("bgSeg");
    clear(bgSeg);
    each(modes, function (item) {
      var btn = ce("button", S.bg.mode === item.key ? "on" : "", item.label);
      btn.type = "button";
      btn.onclick = function () {
        S.bg.mode = item.key;
        renderScene();
        touch();
      };
      bgSeg.appendChild(btn);
    });

    show($("bgBuiltinNote"), S.bg.mode === "hdri");
    show($("bgFileWrap"), S.bg.mode === "upload");
    show($("bgLibraryWrap"), S.bg.mode === "library");
    var cats = $("bgCats");
    if (cats) {
      clear(cats);
      if (!BACKDROPS.length) {
        cats.appendChild(ce("p", "note", "背景库未加载：后端未连接或 assets/backdrops 为空。"));
      } else {
        /* 折叠树：域 → 类别 → 单张照片。可以只看类别（按种子轮换），
           也可以点开类别挑具体一张（那样每次都用它）。 */
        each(["air", "land", "sea"], function (dom) {
          var group = [];
          each(BACKDROPS, function (c) { if ((c.domain || "land") === dom) { group.push(c); } });
          if (!group.length) { return; }
          var total = 0;
          each(group, function (c) { total += c.count; });
          var open = S.bgUi.domains[dom] !== false;
          var sec = ce("div", "bg-sec", "");
          var head = ce("button", "bg-sec-head", "");
          head.type = "button";
          head.appendChild(ce("span", "bg-caret", open ? "▾" : "▸"));
          head.appendChild(ce("span", "", (group[0].domain_label || dom) + " · " + total + " 张"));
          head.onclick = function () {
            S.bgUi.domains[dom] = !open;
            renderScene();
          };
          sec.appendChild(head);
          if (open) {
            var body = ce("div", "bg-sec-body", "");
            each(group, function (c) { body.appendChild(backdropCategoryRow(c)); });
            sec.appendChild(body);
          }
          cats.appendChild(sec);
        });
      }
      renderPlace();
      var note = $("bgLibNote");
      if (note) {
        var cur = null;
        each(BACKDROPS, function (x) { if (x.key === S.bg.category) { cur = x; } });
        if (S.bg.file) {
          note.textContent = "已选定：" + S.bg.file + "（每次都用这张）";
        } else if (cur) {
          note.textContent = "当前：" + cur.label + "，" + cur.count + " 张；未选具体照片，",
          note.textContent += "按种子轮换。点开类别可指定某一张。";
        } else {
          note.textContent = "展开类别可指定具体照片。";
        }
      }
    }
    var path = $("bgPath");
    if (path && document.activeElement !== path) { path.value = S.bg.path; }
    var thumb = $("bgThumb");
    if (thumb) {
      thumb.style.backgroundImage = S.bg.dataUrl ? 'url("' + S.bg.dataUrl + '")' : "";
    }
  }

  /* ------------------------------------------------- 背景图放置 / 轨迹 */
  /* 背景库：一个类别一行，点开看该类全部照片，点某张即选定。 */
  function backdropCategoryRow(c) {
    var row = ce("div", "bg-catrow", "");
    var expanded = S.bgUi.open === c.key;
    var active = S.bg.category === c.key;
    var head = ce("button", "bg-cathead" + (active ? " on" : ""), "");
    head.type = "button";
    head.appendChild(ce("span", "bg-caret", expanded ? "▾" : "▸"));
    if (c.cover) {
      var im = ce("img", "bg-cat-cover");
      im.src = S.bg.category === c.key && S.bg.file
        ? "/api/backdrop/" + c.key + "/" + encodeURIComponent(S.bg.file)
        : c.cover;
      im.alt = "";
      im.loading = "lazy";
      head.appendChild(im);
    }
    head.appendChild(ce("span", "bg-catname", c.label));
    head.appendChild(ce("span", "bg-catcount", c.count + " 张"));
    head.onclick = function () {
      S.bgUi.open = expanded ? "" : c.key;
      if (!active) { S.bg.category = c.key; S.bg.file = ""; }
      renderScene();
      touch();
    };
    row.appendChild(head);
    if (expanded) {
      var grid = ce("div", "bg-thumbs", "");
      var files = c.files || [];
      /* 第一块是"不指定"：回到按种子轮换。 */
      var any = ce("button", "bg-thumb bg-thumb-any" + (active && !S.bg.file ? " on" : ""), "");
      any.type = "button";
      any.appendChild(ce("span", "", "全部/轮换"));
      any.onclick = function () {
        S.bg.category = c.key;
        S.bg.file = "";
        renderScene();
        touch();
      };
      grid.appendChild(any);
      each(files, function (f) {
        var b = ce("button", "bg-thumb" + (S.bg.file === f ? " on" : ""), "");
        b.type = "button";
        b.title = f;
        var t = ce("img", "");
        t.src = "/api/backdrop/" + c.key + "/" + encodeURIComponent(f);
        t.alt = "";
        t.loading = "lazy";
        b.appendChild(t);
        b.onclick = function () {
          S.bg.category = c.key;
          S.bg.file = f;
          renderScene();
          touch();
        };
        grid.appendChild(b);
      });
      row.appendChild(grid);
    }
    return row;
  }

  /* ------------------------------------------------- 多目标（自定义模式） */
  var TARGET_COLORS = ["#ffd166", "#7ee0a6", "#8ab4ff", "#ff9ec4", "#c6a8ff",
                       "#7fe6e0", "#ffb86b", "#b6e388"];

  function targetCount() {
    return Math.max(1, Math.min(8, intval(S.obj.count, 1)));
  }

  /* 把 targets 数组补齐/截断到当前目标数量，缺的用当前选择继承。 */
  function syncTargets() {
    var n = targetCount();
    while (S.targets.length < n) {
      S.targets.push({
        asset: trim(S.obj.asset),
        cls: trim(S.obj.cls) || "fighter",
        uv: null,
        points: [],
        placed: false
      });
    }
    if (S.targets.length > n) { S.targets.length = n; }
    if (S.targetIdx >= n) { S.targetIdx = 0; }
    return S.targets;
  }

  function activeTarget() {
    syncTargets();
    return S.targets[S.targetIdx] || S.targets[0];
  }

  /* 自定义模式：选了背景库且至少有一个目标被放置过。 */
  function isCustomMode() {
    if (S.bg.mode !== "library") { return false; }
    for (var i = 0; i < S.targets.length; i++) {
      if (S.targets[i] && S.targets[i].placed) { return true; }
    }
    return false;
  }

  function placeCoverUrl() {
    if (S.bg.file) {
      return "/api/backdrop/" + S.bg.category + "/" + encodeURIComponent(S.bg.file);
    }
    var cur = null;
    each(BACKDROPS, function (c) { if (c.key === S.bg.category) { cur = c; } });
    return cur ? cur.cover : "";
  }

  /* 目标标签页：切到哪个目标，画布就编辑哪个。 */
  /* 轨迹画完后算一下需要多少帧，用户按需采纳。 */
  function suggestFrames() {
    var spd = num(S.place.speed, 80);
    if (spd <= 0) { return 0; }
    var fps = num(S.capture.fps, 10);
    var longest = 0;
    each(S.targets, function (t) {
      if (!t.points || t.points.length < 2) { return; }
      var L = 0;
      for (var i = 1; i < t.points.length; i++) {
        var dx = (t.points[i][0] - t.points[i - 1][0]) * 600;
        var dy = (t.points[i][1] - t.points[i - 1][1]) * 600 * 0.75;
        L += Math.sqrt(dx * dx + dy * dy);
      }
      if (L > longest) { longest = L; }
    });
    if (longest <= 0) { return 0; }
    return Math.max(2, Math.ceil(longest / spd * fps));
  }

  function renderTargetTabs() {
    var box = $("targetTabs");
    if (!box) { return; }
    clear(box);
    syncTargets();
    if (!isCustomMode() && S.targets.length <= 1) { return; }
    each(S.targets, function (t, i) {
      var col = TARGET_COLORS[i % TARGET_COLORS.length];
      var b = ce("button", "tgt-tab" + (i === S.targetIdx ? " on" : ""), "");
      b.type = "button";
      var sw = ce("span", "tgt-swatch");
      sw.style.background = col;
      b.appendChild(sw);
      var nm = t.asset ? t.asset.split("/").pop().replace(/\.obj$/i, "") : "未选";
      b.appendChild(ce("span", "", "目标" + (i + 1) + " · " + nm));
      if (t.uv) { b.appendChild(ce("span", "tgt-ok", "●")); }
      b.onclick = function () {
        S.targetIdx = i;
        var cur = S.targets[i];
        S.place.u = cur.uv ? cur.uv[0] : 0.5;
        S.place.v = cur.uv ? cur.uv[1] : 0.5;
        S.place.points = (cur.points || []).slice(0);
        renderPlace();
        renderTargetAsset();
      };
      box.appendChild(b);
    });
  }

  /* 当前目标的类别/模型选择。 */
  function renderTargetAsset() {
    var wrap = $("targetAsset");
    if (!wrap) { return; }
    clear(wrap);
    if (!isCustomMode() && S.targets.length <= 1) { return; }
    var t = activeTarget();
    var lab = ce("span", "label", "目标" + (S.targetIdx + 1) + " 型号");
    wrap.appendChild(lab);
    var sel = ce("select", "");
    each(ASSETS, function (a) {
      var o = ce("option", "", a.category + " / " + a.name);
      o.value = a.path;
      if (a.path === t.asset) { o.selected = true; }
      sel.appendChild(o);
    });
    sel.onchange = function () {
      var a = null;
      each(ASSETS, function (x) { if (x.path === sel.value) { a = x; } });
      t.asset = sel.value;
      t.cls = a ? a.cls : t.cls;
      renderPlace();
      touch();
    };
    wrap.appendChild(sel);
  }

  function renderPlace() {
    var wrap = $("placeWrap");
    if (!wrap) { return; }
    var on = S.bg.mode === "library";
    show(wrap, on);
    if (!on) { return; }
    renderTargetTabs();
    renderTargetAsset();
    var img = $("placeImg");
    var cover = placeCoverUrl();
    if (img && cover && img.getAttribute("src") !== cover) { img.src = cover; }
    var seg = $("placeModeSeg");
    if (seg) {
      clear(seg);
      each([{ key: "place", label: "放置目标" }, { key: "path", label: "画轨迹" }], function (m) {
        var b = ce("button", S.place.mode === m.key ? "on" : "", m.label);
        b.type = "button";
        b.onclick = function () { S.place.mode = m.key; renderPlace(); };
        seg.appendChild(b);
      });
    }
    var marks = $("placeMarkers");
    if (marks) {
      clear(marks);
      syncTargets();
      /* 一次画出所有目标：位置点 + 轨迹折线，当前编辑的加粗高亮。 */
      each(S.targets, function (t, ti) {
        var col = TARGET_COLORS[ti % TARGET_COLORS.length];
        var act = ti === S.targetIdx;
        if (t.uv) {
          var dot = ce("div", "place-dot" + (act ? " first" : ""));
          dot.style.left = (t.uv[0] * 100) + "%";
          dot.style.top = (t.uv[1] * 100) + "%";
          dot.style.borderColor = col;
          if (act) { dot.style.background = col; }
          dot.title = "目标 " + (ti + 1);
          marks.appendChild(dot);
        }
        if (t.points && t.points.length > 1) {
          for (var i = 0; i < t.points.length; i++) {
            var p = t.points[i];
            var d2 = ce("div", "place-dot");
            d2.style.left = (p[0] * 100) + "%";
            d2.style.top = (p[1] * 100) + "%";
            d2.style.borderColor = col;
            d2.style.opacity = act ? "1" : "0.55";
            d2.style.width = act ? "8px" : "6px";
            d2.style.height = act ? "8px" : "6px";
            marks.appendChild(d2);
          }
        }
      });
    }
    var hint = $("placeHint");
    if (hint) {
      if (S.place.mode === "place") {
        hint.textContent = "点一下图面即把「目标" + (S.targetIdx + 1) + "」放到该位置。";
      } else {
        var n = suggestFrames();
        hint.textContent = "依次点击为「目标" + (S.targetIdx + 1) + "」画轨迹（当前 "
          + S.place.points.length + " 个点）"
          + (n ? "；按当前速度约需 " + n + " 帧" : "") + "。";
      }
    }
    var status = $("placeStatus");
    if (status) { status.textContent = S.place.busy ? "渲染中…" : ""; }
    var out = $("placeOut");
    if (out) {
      show(out, !!S.place.url);
      var pv = $("placePreviewImg");
      if (pv && S.place.url) { pv.src = S.place.url; }
    }
  }

  function placeClick(ev) {
    var box = $("placeCanvas");
    if (!box) { return; }
    var rect = box.getBoundingClientRect();
    if (!rect.width || !rect.height) { return; }
    var u = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
    var v = Math.max(0, Math.min(1, (ev.clientY - rect.top) / rect.height));
    var t = activeTarget();
    if (S.place.mode === "place") {
      t.uv = [u, v];
      t.points = [];
      t.placed = true;
    } else {
      if (!t.uv) { t.uv = [u, v]; t.placed = true; }
      t.points.push([u, v]);
    }
    S.place.u = t.uv ? t.uv[0] : u;
    S.place.v = t.uv ? t.uv[1] : v;
    S.place.points = t.points.slice(0);
    S.place.placed = isCustomMode();
    renderPlace();
    touch();
  }

  function placePreview() {
    if (S.place.busy) { return; }
    var u = S.place.points.length ? S.place.points[0][0] : S.place.u;
    var v = S.place.points.length ? S.place.points[0][1] : S.place.v;
    var st = $("placeStatus");
    S.place.busy = true;
    renderPlace();
    /* 发送与「生成」完全相同的表单：预览必须走同一条 compose 路径，
       否则目标数、轨迹、环境类型都可能和实际生成不一致。 */
    var payload = buildPayload();
    payload.width = 320;
    payload.samples = 12;
    ajax("POST", API + "/place", payload, function (err, d) {
      S.place.busy = false;
      if (err) {
        if (st) { st.textContent = "预览失败：" + (err.error || err.message || "未知错误"); }
        renderPlace();
        return;
      }
      S.place.url = d.url + "?t=" + Date.now();
      if (st) {
        st.textContent = (d.targets || 1) + " 个目标 · " + d.sensor
          + " · " + (d.size || []).join("×") + " · " + d.seconds + " 秒";
      }
      renderPlace();
    });
  }

  /* ------------------------------------------------------------ 渲染：目标 */

  function assetMeta(asset) {
    if (!asset) { return ""; }
    var size = asset.size_m || [0, 0, 0];
    var dims = [];
    each(size, function (v) { dims.push((Math.round(num(v, 0) * 10) / 10).toFixed(1)); });
    var text = dims.join(" × ") + " m · " + (asset.tris || 0).toLocaleString() + " 三角面 · " + asset.cls;
    var px = predictedPixels(asset);
    if (px) { text += " · 目标约 " + px + " px"; }
    return text;
  }

  function predictedPixels(asset) {
    if (!asset) { return ""; }
    var dims = asset.size_m || [];
    var size = 0;
    each(dims, function (v) { size = Math.max(size, num(v, 0)); });
    size = size * num(S.obj.scale, 1);
    var width = (OPT.views && OPT.views.narrow && OPT.views.narrow.width) || 960;
    var fov = num(S.view.fov_deg, 2.2);
    var km = num(S.view.range_km, 8);
    if (!(size > 0) || !(km > 0) || !(fov > 0)) { return ""; }
    var focal = width / (2 * Math.tan(fov * Math.PI / 360));
    var near = size * focal / (km * 1000 * 0.85);
    var far = size * focal / (km * 1000 * 1.15);
    function round(v) { return Math.max(1, Math.round(v)); }
    return round(far) + "–" + round(near);
  }

  function renderSizeSeg() {
    var host = $("sizeSeg");
    clear(host);
    each(["small", "medium", "large"], function (key) {
      var btn = ce("button", S.size === key ? "on" : "", SIZE_LABEL[key]);
      btn.type = "button";
      btn.setAttribute("data-v", key);
      btn.onclick = function () {
        S.size = key;
        S.obj.scale = SCALE[key];
        var scale = $("objScale");
        if (scale) { scale.value = S.obj.scale; }
        renderSizeSeg();
        refreshAssetMeta();
        renderSummaries();
        touch();
      };
      host.appendChild(btn);
    });
  }

  function refreshAssetMeta() {
    var host = $("assetMeta");
    if (!host) { return; }
    clear(host);
    var asset = findAsset(S.obj.asset);
    var text = asset ? assetMeta(asset) : (trim(S.obj.asset) ? "手动路径：" + S.obj.asset : "");
    if (text) {
      host.appendChild(ce("span", "", text));
      if (asset) {
        host.appendChild(ce("span", "prov",
          "目标像素按 range_km 与 fov 反算，低于 1 px 无法标注。"));
      }
    }
  }

  function renderTarget() {
    var pickers = $("assetPickers");
    var state = $("assetState");
    var hasAssets = ASSETS.length > 0;
    show(pickers, hasAssets);

    if (!hasAssets) {
      show(state, true);
      state.className = "notice";
      clear(state);
      var message = ASSET_ERROR
        ? "资产库暂不可用：" + ASSET_ERROR
        : "资产库暂不可用：assets/models 下没有扫描到模型。";
      state.appendChild(document.createTextNode(message + " "));
      state.appendChild(document.createTextNode("仍可在下方「高级」里手填模型路径。"));
    } else if (ASSET_ERROR) {
      show(state, true);
      state.className = "notice";
      state.textContent = "资产库只加载了一部分：" + ASSET_ERROR;
    } else {
      show(state, false);
      state.textContent = "";
    }

    if (hasAssets) {
      var cats = [];
      var byCat = {};
      each(ASSETS, function (asset) {
        if (!byCat[asset.category]) {
          byCat[asset.category] = [];
          cats.push(asset.category);
        }
        byCat[asset.category].push(asset);
      });
      if (indexOf(cats, S.obj.category) < 0) { S.obj.category = cats[0]; }
      if (!findAsset(S.obj.asset) || findAsset(S.obj.asset).category !== S.obj.category) {
        var first = byCat[S.obj.category][0];
        S.obj.asset = first.path;
        S.obj.cls = first.cls;
      }

      var catSel = $("catSel");
      clear(catSel);
      each(cats, function (cat) {
        var opt = ce("option", "", cat + "（" + byCat[cat].length + "）");
        opt.value = cat;
        catSel.appendChild(opt);
      });
      catSel.value = S.obj.category;
      catSel.onchange = function () {
        S.obj.category = catSel.value;
        var list = byCat[catSel.value] || [];
        if (list.length) {
          S.obj.asset = list[0].path;
          S.obj.cls = list[0].cls;
        }
        renderTarget();
        touch();
      };

      var assetSel = $("assetSel");
      clear(assetSel);
      each(byCat[S.obj.category] || [], function (asset) {
        var opt = ce("option", "", asset.name);
        opt.value = asset.path;
        assetSel.appendChild(opt);
      });
      assetSel.value = S.obj.asset;
      assetSel.onchange = function () {
        var asset = findAsset(assetSel.value);
        S.obj.asset = assetSel.value;
        if (asset) { S.obj.cls = asset.cls; }
        renderTarget();
        touch();
      };
    }

    renderSizeSeg();
    var count = $("objCount");
    if (count) { count.value = S.obj.count; }
    var scale = $("objScale");
    if (scale) { scale.value = S.obj.scale; }
    var cls = $("objCls");
    if (cls && document.activeElement !== cls) { cls.value = S.obj.cls; }
    var path = $("objPath");
    if (path && document.activeElement !== path) { path.value = S.obj.asset; }
    refreshAssetMeta();
  }

  /* ------------------------------------------------------------ 渲染：干扰 */

  function newEntity(kind) {
    return { kind: kind, params: {}, extra: { id: "ecm_" + kind } };
  }

  function entityByKind(kind) {
    for (var i = 0; i < S.ecm.entities.length; i++) {
      if (S.ecm.entities[i].kind === kind) { return S.ecm.entities[i]; }
    }
    return null;
  }

  function entityValue(entity, key) {
    if (entity && has(entity.params, key)) { return entity.params[key]; }
    var defaults = ecmDefaults(entity.kind);
    return has(defaults, key) ? defaults[key] : null;
  }

  function paramProv(kind, key) {
    if (key === "erp_w") { return "Ku 波段 500 m 下 200 W ≈ J/S +10 dB（实测标定）。"; }
    if (key === "jam_bandwidth_mhz") { return "压制式按接收带宽摊薄，需 ~B_jam/B_rx 倍功率。"; }
    if (key === "sidelobe_db") { return "-25 dB：干扰机不指雷达时的典型旁瓣电平。"; }
    if (key === "size_m" && kind === "corner_reflector") {
      return "0.3 m 三面角在 Ku 约 28.8 dBsm，按设计就会压过目标。";
    }
    if (key === "decay_s") { return "RCS ∝ (r0/r)^2 稀释，叠加沉降指数。"; }
    if (key === "mat") { return "材质名由 simkit 材质表解析。"; }
    if (key === "look_at_sensor") { return "开启后干扰波束始终指向本机传感器。"; }
    if (key === "mode") { return "阻塞式覆盖整个接收带宽；瞄准式功率更集中。"; }
    if (key === "count") { return "假目标个数与步进共同决定距离门上的峰数。"; }
    return "";
  }

  function scalarParam(entity, key, value) {
    var wrap = ce("div", "param");
    wrap.appendChild(ce("label", "label", paramLabel(key)));
    if (typeof value === "boolean") {
      var btn = ce("button", "chip" + (value ? " on" : ""), value ? "开启" : "关闭");
      btn.type = "button";
      btn.onclick = function () {
        var current = has(entity.params, key) ? !!entity.params[key] : !!value;
        entity.params[key] = !current;
        btn.className = "chip" + (entity.params[key] ? " on" : "");
        btn.textContent = entity.params[key] ? "开启" : "关闭";
        touch();
      };
      wrap.appendChild(btn);
    } else if (key === "mode" && typeof value === "string") {
      var sel = ce("select", "");
      each([{ key: "barrage", label: "阻塞式" }, { key: "spot", label: "瞄准式" }], function (item) {
        var opt = ce("option", "", item.label);
        opt.value = item.key;
        sel.appendChild(opt);
      });
      sel.value = String(value);
      sel.onchange = function () {
        entity.params[key] = sel.value;
        touch();
      };
      wrap.appendChild(sel);
    } else if (typeof value === "number") {
      var input = ce("input", "");
      input.type = "number";
      input.step = "any";
      input.value = value;
      input.oninput = function () {
        var v = parseFloat(input.value);
        if (!isFinite(v)) { return; }
        entity.params[key] = v;
        touch();
      };
      wrap.appendChild(input);
    } else {
      var text = ce("input", "");
      text.type = "text";
      text.value = value === null || value === undefined ? "" : String(value);
      text.oninput = function () {
        entity.params[key] = text.value;
        touch();
      };
      wrap.appendChild(text);
    }
    var prov = paramProv(entity.kind, key);
    if (prov) { wrap.appendChild(ce("span", "prov", prov)); }
    return wrap;
  }

  function vectorParam(entity, key, value) {
    var wrap = ce("div", "param wide");
    wrap.appendChild(ce("label", "label", paramLabel(key)));
    var row = ce("div", "vec3");
    var arr = isArray(value) ? value.slice(0, 3) : [0, 0, 0];
    while (arr.length < 3) { arr.push(0); }
    each(arr, function (initial, index) {
      var input = ce("input", "");
      input.type = "number";
      input.step = "any";
      input.value = initial;
      input.oninput = function () {
        var v = parseFloat(input.value);
        if (!isFinite(v)) { return; }
        var current = isArray(entity.params[key]) ? entity.params[key].slice(0, 3)
          : (isArray(ecmDefaults(entity.kind)[key]) ? ecmDefaults(entity.kind)[key].slice(0, 3) : [0, 0, 0]);
        while (current.length < 3) { current.push(0); }
        current[index] = v;
        entity.params[key] = current;
        touch();
      };
      row.appendChild(input);
    });
    wrap.appendChild(row);
    if (key === "size") {
      wrap.appendChild(ce("span", "prov", "按模型包围盒自动取值；不是实测尺寸。"));
    } else if (key === "wind") {
      wrap.appendChild(ce("span", "prov", "默认 0 m/s；箔条云会随风漂移。"));
    } else if (key === "aspect") {
      wrap.appendChild(ce("span", "prov", "默认 (1,1,1)；大于 1 表示椭球在该轴拉长。"));
    }
    return wrap;
  }

  function removeEntityAt(index) {
    S.ecm.entities.splice(index, 1);
    renderEcm();
    renderSummaries();
    touch();
  }

  function toggleEcmKind(kind, checked) {
    if (checked && !entityByKind(kind)) {
      S.ecm.entities.push(newEntity(kind));
    } else if (!checked) {
      var next = [];
      each(S.ecm.entities, function (entity) {
        if (entity.kind !== kind) { next.push(entity); }
      });
      S.ecm.entities = next;
    }
    renderEcm();
    renderSummaries();
    touch();
  }

  function renderEntityBlock(entity, index) {
    var kind = ecmKind(entity.kind);
    var block = ce("div", "ecm-block");
    var head = ce("div", "ecm-block-head");
    var title = ce("div", "ecm-block-title", kind ? kind.label : entity.kind);
    if (kind) { title.appendChild(ce("em", "", ecmGroupLabel(kind.group))); }
    head.appendChild(title);
    var remove = ce("button", "ecm-remove", "×");
    remove.type = "button";
    remove.title = "移除这个实体";
    remove.onclick = function () { removeEntityAt(index); };
    head.appendChild(remove);
    block.appendChild(head);

    var defaults = ecmDefaults(entity.kind);
    var keys = Object.keys(defaults);
    var scalarKeys = [];
    var vectorKeys = [];
    var i;
    for (i = 0; i < keys.length; i++) {
      var value = entityValue(entity, keys[i]);
      if (isArray(value)) { vectorKeys.push(keys[i]); }
      else if (value !== null && value !== undefined) { scalarKeys.push(keys[i]); }
    }
    if (!scalarKeys.length && !vectorKeys.length) {
      block.appendChild(ce("p", "entity-note", "这个类型的参数表尚未加载。"));
      return block;
    }
    var grid = ce("div", "param-grid");
    each(scalarKeys, function (key) {
      grid.appendChild(scalarParam(entity, key, entityValue(entity, key)));
    });
    block.appendChild(grid);
    block.appendChild(ce("p", "entity-note",
      "未标出处的项是物理层近似默认值，不是实测；提交时会带上该类型的完整参数。"));
    if (vectorKeys.length) {
      var adv = ce("details", "adv");
      adv.appendChild(ce("summary", "", "高级 · 向量与材质"));
      var body = ce("div", "adv-body");
      var vgrid = ce("div", "param-grid");
      each(vectorKeys, function (key) {
        vgrid.appendChild(vectorParam(entity, key, entityValue(entity, key)));
      });
      body.appendChild(vgrid);
      adv.appendChild(body);
      block.appendChild(adv);
    }
    return block;
  }

  function renderAugment() {
    /* 图像级干扰增广。和上面的物理干扰实体分开：实体进 3D 场景（能被光学/红外
       看到），增广只改图像像素。 */
    var sel = $("augmentSel");
    var hint = $("augmentHint");
    if (!sel || !hint) { return; }
    sel.value = S.interferenceMode || "inherit";
    if (!sel.dataset.bound) {
      sel.dataset.bound = "1";
      sel.onchange = function () {
        S.interferenceMode = sel.value;
        hint.textContent = AUGMENT_HINT[sel.value] || "";
        renderSummary();
      };
    }
    hint.textContent = AUGMENT_HINT[sel.value] || "";
  }

  function renderEcm() {
    var picks = $("ecmPicks");
    var blocks = $("ecmBlocks");
    var hint = $("ecmHint");
    if (!picks || !blocks) { return; }
    renderAugment();
    clear(picks);
    clear(blocks);
    var kinds = (OPT.ecm && OPT.ecm.kinds) || [];
    if (!kinds.length) {
      hint.textContent = "干扰类型表尚未加载：后端未连接或 /api/ecm-kinds 为空。";
      return;
    }
    each([{ key: "active", label: "有源" }, { key: "passive", label: "无源" }], function (group) {
      var inGroup = [];
      each(kinds, function (kind) {
        if ((kind.group || "active") === group.key) { inGroup.push(kind); }
      });
      if (!inGroup.length) { return; }
      picks.appendChild(ce("div", "ecm-group-label", group.label));
      each(inGroup, function (kind) {
        var checked = !!entityByKind(kind.key);
        var row = ce("label", "ecm-pick" + (checked ? " on" : ""));
        var input = ce("input", "");
        input.type = "checkbox";
        input.checked = checked;
        input.onchange = function () { toggleEcmKind(kind.key, input.checked); };
        row.appendChild(input);
        row.appendChild(ce("span", "", kind.label));
        picks.appendChild(row);
      });
    });
    hint.textContent = S.ecm.entities.length
      ? "当前模式：" + (MODE_LABEL[ecmMode()] || ecmMode()) + "，共 " + S.ecm.entities.length + " 个实体。"
      : "勾选即添加；不勾选则本次无干扰。模式会按实体自动推断。";
    each(S.ecm.entities, function (entity, index) {
      if (index > 0) { blocks.appendChild(ce("div", "ecm-divider")); }
      blocks.appendChild(renderEntityBlock(entity, index));
    });
  }

  /* ------------------------------------------------------------ 渲染：视角 */

  function setFov(fov) {
    S.view.fov_deg = clamp(num(fov, 2.2), 0.1, 120);
    var input = $("fovDeg");
    if (input && document.activeElement !== input) { input.value = S.view.fov_deg; }
    refreshAssetMeta();
  }

  function trajToPx(point) {
    var half = 240;
    var step = 30;             /* 480 px 画布 = 16 km，1 km = 30 px */
    return [half + num(point[0], 0) / 1000 * step,
            half - num(point[1], 0) / 1000 * step];
  }

  function trajFromPx(px, py) {
    var half = 240;
    var step = 30;
    var x = Math.round((px - half) / step * 1000 / 10) * 10;
    var y = Math.round((half - py) / step * 1000 / 10) * 10;
    return [clamp(x, -7900, 7900), clamp(y, -7900, 7900)];
  }

  function drawTrajCanvas() {
    var canvas = $("trajCanvas");
    if (!canvas || !canvas.getContext) { return; }
    var ctx = canvas.getContext("2d");
    if (!ctx) { return; }
    var W = canvas.width;
    var H = canvas.height;
    var cx = W / 2;
    var cy = H / 2;
    var step = 30;
    var i;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#080a0e";
    ctx.fillRect(0, 0, W, H);

    ctx.lineWidth = 1;
    for (i = -8; i <= 8; i++) {
      ctx.strokeStyle = i === 0 ? "rgba(255,255,255,0.16)" : "rgba(255,255,255,0.055)";
      ctx.beginPath();
      ctx.moveTo(cx + i * step, 0);
      ctx.lineTo(cx + i * step, H);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, cy + i * step);
      ctx.lineTo(W, cy + i * step);
      ctx.stroke();
    }

    ctx.fillStyle = "rgba(255,255,255,0.32)";
    ctx.font = '10px -apple-system, "Microsoft YaHei", sans-serif';
    ctx.fillText("X", W - 14, cy - 7);
    ctx.fillText("Y", cx + 7, 13);

    /* 比例尺：1 km = 30 px */
    ctx.strokeStyle = "rgba(255,255,255,0.4)";
    ctx.beginPath();
    ctx.moveTo(14, H - 24);
    ctx.lineTo(44, H - 24);
    ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.38)";
    ctx.fillText("1 km", 14, H - 10);

    /* 相机在中心 */
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#5ea8ff";
    ctx.fill();
    ctx.strokeStyle = "#0b0e13";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.fillText("相机", cx + 9, cy + 4);

    var pts = S.traj.points;
    var smooth = S.traj.smoothed;

    if (pts.length > 1) {
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "rgba(94,168,255,0.35)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      each(pts, function (point, index) {
        var q = trajToPx(point);
        if (index === 0) { ctx.moveTo(q[0], q[1]); }
        else { ctx.lineTo(q[0], q[1]); }
      });
      ctx.stroke();
      ctx.restore();
    }
    each(pts, function (point) {
      var q = trajToPx(point);
      ctx.beginPath();
      ctx.arc(q[0], q[1], 3, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(94,168,255,0.55)";
      ctx.fill();
    });

    if (isArray(smooth) && smooth.length > 1) {
      ctx.strokeStyle = "#5ea8ff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      each(smooth, function (point, index) {
        var q = trajToPx(point);
        if (index === 0) { ctx.moveTo(q[0], q[1]); }
        else { ctx.lineTo(q[0], q[1]); }
      });
      ctx.stroke();
      var first = trajToPx(smooth[0]);
      var last = trajToPx(smooth[smooth.length - 1]);
      each([first, last], function (q) {
        ctx.beginPath();
        ctx.arc(q[0], q[1], 3.5, 0, Math.PI * 2);
        ctx.fillStyle = "#dcebff";
        ctx.fill();
      });
    }
  }

  function renderTrajState() {
    var drawn = S.traj.mode === "drawn";
    show($("fixedView"), !drawn);
    show($("drawnView"), drawn);
    var note = $("trajModeNote");
    if (note) {
      note.textContent = drawn
        ? "左键加点，右键撤销；生成只使用优化后的平滑曲线。"
        : "固定轨迹使用「距离 + 方位」生成标准迎头/追击航路。";
    }
    if (!drawn) { drawTrajCanvas(); return; }

    var alt = $("trajAlt");
    if (alt && document.activeElement !== alt) { alt.value = S.traj.alt_m; }
    var speed = $("trajSpeed");
    if (speed && document.activeElement !== speed) { speed.value = S.traj.speed_mps; }
    var closed = $("trajClosed");
    if (closed) {
      closed.className = "chip" + (S.traj.closed ? " on" : "");
      closed.textContent = S.traj.closed ? "是" : "否";
    }

    var stats = $("trajStats");
    if (stats) {
      if (!S.traj.points.length) {
        stats.textContent = "在画布上点至少两个点。";
      } else if (!S.traj.smoothed) {
        stats.textContent = "已选 " + S.traj.points.length + " 个点；点「优化轨迹」生成可飞曲线。";
      } else if (S.traj.stats) {
        stats.textContent = fmtTrajStats(S.traj.stats);
      } else {
        stats.textContent = "已载入手绘轨迹 · " +
          (polylineLengthM(S.traj.smoothed) / 1000).toFixed(1) + " km · 速度 " +
          (Math.round(num(S.traj.speed_mps, 0) * 10) / 10) + " m/s（平滑点）。";
      }
    }
    var warn = $("trajWarn");
    if (warn) {
      warn.className = "traj-warn";
      if (S.traj.error) {
        warn.className = "traj-warn bad";
        warn.textContent = S.traj.error;
      } else if (S.traj.warnings && S.traj.warnings.length) {
        warn.textContent = S.traj.warnings.join("；");
      } else {
        warn.textContent = "";
      }
    }
    var btn = $("trajOptimize");
    if (btn) {
      btn.disabled = S.traj.optimizing || S.traj.points.length < 2;
      btn.textContent = S.traj.optimizing ? "优化中…" : "优化轨迹";
    }
    drawTrajCanvas();
  }

  function polylineLengthM(points) {
    var total = 0;
    for (var i = 1; i < points.length; i++) {
      var a = points[i - 1];
      var b = points[i];
      var dx = num(b[0], 0) - num(a[0], 0);
      var dy = num(b[1], 0) - num(a[1], 0);
      var dz = num(b[2], 0) - num(a[2], 0);
      total += Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    return total;
  }

  function fmtTrajStats(stats) {
    var km = num(stats.length_m, 0) / 1000;
    var dur = num(stats.duration_s, 0);
    var g = num(stats.max_g, 0);
    var bank = num(stats.max_bank_deg, 0);
    function trimNum(v, digits) {
      var n = Math.round(v * Math.pow(10, digits)) / Math.pow(10, digits);
      return String(n);
    }
    return "长度 " + km.toFixed(1) + " km · 时长 " + trimNum(dur, 1) +
      " s · 最大过载 " + trimNum(g, 2) + " g · 最大坡度 " + trimNum(bank, 1) + "°";
  }

  function invalidateTrajectory() {
    S.traj.smoothed = null;
    S.traj.stats = null;
    S.traj.warnings = [];
    S.traj.error = "";
  }

  function addTrajPointFromEvent(ev) {
    var canvas = $("trajCanvas");
    if (!canvas) { return; }
    var rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) { return; }
    var px = (ev.clientX - rect.left) * canvas.width / rect.width;
    var py = (ev.clientY - rect.top) * canvas.height / rect.height;
    var point = trajFromPx(px, py);
    var pts = S.traj.points;
    if (pts.length) {
      var last = pts[pts.length - 1];
      if (Math.abs(last[0] - point[0]) < 5 && Math.abs(last[1] - point[1]) < 5) { return; }
    }
    pts.push(point);
    invalidateTrajectory();
    renderTrajState();
    renderSummaries();
    touch();
  }

  function undoTrajPoint() {
    if (!S.traj.points.length) { return; }
    S.traj.points.pop();
    invalidateTrajectory();
    renderTrajState();
    renderSummaries();
    touch();
  }

  function clearTrajectory() {
    S.traj.points = [];
    invalidateTrajectory();
    renderTrajState();
    renderSummaries();
    touch();
  }

  function optimizeTrajectory() {
    if (S.traj.optimizing || S.traj.points.length < 2) { return; }
    S.traj.optimizing = true;
    S.traj.error = "";
    renderTrajState();
    ajax("POST", API + "/trajectory/smooth", {
      points: S.traj.points.slice(0),
      alt_m: num(S.traj.alt_m, 2500),
      speed_mps: num(S.traj.speed_mps, 250),
      closed: !!S.traj.closed
    }, function (err, data) {
      S.traj.optimizing = false;
      if (err) {
        S.traj.error = err.error || "轨迹优化失败";
        S.traj.smoothed = null;
        S.traj.stats = null;
        S.traj.warnings = [];
      } else if (!data || data.ok === false || !isArray(data.points) || data.points.length < 2) {
        S.traj.error = (data && data.error) || "后端没有返回有效的平滑轨迹";
        S.traj.smoothed = null;
        S.traj.stats = null;
        S.traj.warnings = [];
      } else {
        S.traj.smoothed = data.points;
        if (num(data.max_speed_mps, 0) > 0) {
          /* 后端可能为了 1.5 g 上限自动降速；配方里必须写实际速度，
             否则生成的航线会按原始速度重新采样，过载又超限。 */
          S.traj.speed_mps = num(data.max_speed_mps, S.traj.speed_mps);
        }
        S.traj.stats = {
          length_m: data.length_m, duration_s: data.duration_s,
          max_g: data.max_g, max_bank_deg: data.max_bank_deg
        };
        S.traj.warnings = data.warnings || [];
        S.traj.error = "";
        toast("轨迹已优化：" + fmtTrajStats(S.traj.stats));
      }
      renderTrajState();
      renderSummaries();
      touch();
    });
  }

  function renderView() {
    var modeSeg = $("trajModeSeg");
    clear(modeSeg);
    each([{ key: "preset", label: "固定（默认）" }, { key: "drawn", label: "手绘轨迹" }], function (item) {
      var btn = ce("button", S.traj.mode === item.key ? "on" : "", item.label);
      btn.type = "button";
      btn.onclick = function () {
        S.traj.mode = item.key;
        renderView();
        renderSummaries();
        touch();
      };
      modeSeg.appendChild(btn);
    });

    var aspect = $("aspectSeg");
    clear(aspect);
    /* 迎头/追击只对空中目标有意义，俯视只对地面与海面目标有意义。不适用的
       那一个置灰，避免选了以后被引擎悄悄换掉、摘要和实际不一致。 */
    var surface = isSurfaceTarget();
    /* 换目标时把不适用的视角自动换成可用的那个：切成坦克后还停在"迎头"的
       话，三个按钮全灰、摘要却写着迎头，实际跑起来又被引擎改成俯视。 */
    if (surface && S.view.aspect !== "look_down") {
      S.view.aspect = "look_down";
    } else if (!surface && S.view.aspect === "look_down") {
      S.view.aspect = "head_on";
    }
    each([{ key: "head_on", label: "迎头", air: true },
          { key: "pursuit", label: "追击", air: true },
          { key: "look_down", label: "俯视", air: false }], function (item) {
      var enabled = surface ? !item.air : item.air;
      var btn = ce("button", S.view.aspect === item.key && enabled ? "on" : "", item.label);
      btn.type = "button";
      if (!enabled) {
        btn.disabled = true;
        btn.title = surface ? "地面/海面目标使用俯视" : "空中目标使用迎头或追击";
      } else {
        btn.onclick = function () {
          S.view.aspect = item.key;
          renderView();
          renderSummaries();
          touch();
        };
      }
      aspect.appendChild(btn);
    });

    var presets = [
      { key: "2.2", label: "2.2°", fov: 2.2, dual: false },
      { key: "8", label: "8°", fov: 8.0, dual: false },
      { key: "18", label: "18°", fov: 18.0, dual: false },
      { key: "dual", label: "双视场", fov: null, dual: true }
    ];
    var fovSeg = $("fovSeg");
    clear(fovSeg);
    each(presets, function (item) {
      var active = item.dual
        ? S.view.views === "dual"
        : (S.view.views !== "dual" && Math.abs(S.view.fov_deg - item.fov) < 1e-6);
      var btn = ce("button", active ? "on" : "", item.label);
      btn.type = "button";
      btn.onclick = function () {
        if (item.dual) {
          S.view.views = "dual";
          if (S.view.fov_deg >= 18) { S.view.fov_deg = 2.2; }
        } else {
          S.view.views = "single";
          S.view.fov_deg = item.fov;
        }
        renderView();
        renderSummaries();
        touch();
      };
      fovSeg.appendChild(btn);
    });

    var views = $("viewsSeg");
    clear(views);
    each([{ key: "single", label: "单视场" }, { key: "dual", label: "双视场" }], function (item) {
      var btn = ce("button", S.view.views === item.key ? "on" : "", item.label);
      btn.type = "button";
      btn.onclick = function () {
        S.view.views = item.key;
        renderView();
        renderSummaries();
        touch();
      };
      views.appendChild(btn);
    });

    var slider = $("rangeSlider");
    var number = $("rangeNum");
    var fov = $("fovDeg");
    if (slider && document.activeElement !== slider) { slider.value = clamp(num(S.view.range_km, 8), 0.1, 100); }
    if (number && document.activeElement !== number) { number.value = S.view.range_km; }
    if (fov && document.activeElement !== fov) { fov.value = S.view.fov_deg; }
    renderTrajState();
    renderPlan();
  }

  /* ------------------------------------------------------------ 渲染：采集 */

  function renderCapture() {
    var host = $("sensorCards");
    clear(host);
    each(OPT.sensors || [], function (sensor) {
      var on = indexOf(S.capture.sensors, sensor.key) >= 0;
      var btn = ce("button", "sensor" + (on ? " on" : ""));
      btn.type = "button";
      btn.appendChild(ce("b", "", sensor.label));
      btn.appendChild(ce("span", "", sensor.note || ""));
      btn.onclick = function () {
        var index = indexOf(S.capture.sensors, sensor.key);
        if (index >= 0) {
          S.capture.sensors.splice(index, 1);
          btn.className = "sensor";
        } else {
          S.capture.sensors.push(sensor.key);
          btn.className = "sensor on";
        }
        renderSummaries();
        touch();
      };
      host.appendChild(btn);
    });
    var frames = $("frames");
    if (frames && document.activeElement !== frames) { frames.value = S.capture.frames; }
    var fps = $("fps");
    if (fps && document.activeElement !== fps) { fps.value = S.capture.fps; }
  }

  /* ------------------------------------------------------------ 渲染：输出 */

  function renderOutput() {
    var host = $("outputToggles");
    clear(host);
    var defs = [
      { key: "preview", label: "拼接预览", desc: "每模态 4 帧联系表，快速看效果" },
      { key: "coco", label: "COCO 标注", desc: "检测框与类别写成标准 JSON" },
      { key: "yolo", label: "YOLO 标签", desc: "归一化 txt，可直接用于训练" },
      { key: "readme", label: "说明文档", desc: "数据集结构、参数与口径" }
    ];
    each(defs, function (item) {
      var on = !!S.output[item.key];
      var btn = ce("button", "toggle" + (on ? " on" : ""));
      btn.type = "button";
      var main = ce("span", "");
      main.appendChild(ce("span", "t-label", item.label));
      main.appendChild(ce("span", "t-desc", item.desc));
      btn.appendChild(main);
      btn.appendChild(ce("span", "switch"));
      btn.onclick = function () {
        S.output[item.key] = !S.output[item.key];
        btn.className = "toggle" + (S.output[item.key] ? " on" : "");
        renderSummaries();
        touch();
      };
      host.appendChild(btn);
    });
  }

  /* ------------------------------------------------------------ 渲染：总装 */

  function renderAll() {
    renderScene();
    renderTarget();
    renderEcm();
    renderView();
    renderCapture();
    renderOutput();
    renderSummaries();
    updateNameHint();

    var dataName = $("dataName");
    if (dataName && document.activeElement !== dataName) { dataName.value = S.name; }
    var seed = $("seed");
    if (seed && document.activeElement !== seed) { seed.value = S.capture.seed; }
    var outInput = $("outputDir");
    if (outInput && document.activeElement !== outInput) { outInput.value = S.outputDir; }
    renderOutStatus();
    updateGenEnabled();
  }

  /* ------------------------------------------------------------ 输出目录 */

  function joinPath(base, name) {
    return String(base || "").replace(/[\/\\]+$/, "") + "/" + name;
  }

  function browseTo(path) {
    OUT.seq++;
    var seq = OUT.seq;
    OUT.checking = true;
    OUT.error = "";
    renderOutStatus();
    ajax("GET", API + "/browse?path=" + encodeURIComponent(path || ""), null,
      function (err, data) {
        if (seq !== OUT.seq) { return; }
        OUT.checking = false;
        if (err) {
          OUT.info = null;
          OUT.error = err.error || "无法浏览目录";
          renderBrowser();
          renderOutStatus();
          return;
        }
        OUT.info = data || null;
        if (OUT.info && OUT.info.writable === false) {
          OUT.localBad = true;
          OUT.localText = "输出目录不可写：" + (OUT.info.requested || OUT.info.path);
        } else {
          OUT.localBad = false;
          OUT.localText = "";
        }
        renderBrowser();
        renderOutStatus();
      });
  }

  function setOutputDir(value, doBrowse) {
    S.outputDir = value;
    var input = $("outputDir");
    if (input) { input.value = value; }
    OUT.info = null;
    OUT.error = "";
    OUT.checking = false;
    OUT.localBad = false;
    OUT.localText = "";
    if (doBrowse && trim(value)) {
      browseTo(value);
    } else {
      renderOutStatus();
    }
  }

  function scheduleBrowse() {
    if (browseTimer) { clearTimeout(browseTimer); }
    browseTimer = setTimeout(function () {
      if (trim(S.outputDir)) { browseTo(S.outputDir); }
      else {
        OUT.info = null;
        OUT.error = "";
        OUT.checking = false;
        OUT.localBad = false;
        OUT.localText = "";
        renderOutStatus();
      }
    }, 450);
  }

  function renderBrowser() {
    var wrap = $("browser");
    if (!wrap) { return; }
    show(wrap, OUT.open);
    if (!OUT.open) { return; }
    var path = $("browserPath");
    var dirs = $("browserDirs");
    var note = $("browserNote");
    clear(dirs);
    if (!OUT.info) {
      path.textContent = "…";
      note.textContent = OUT.error || "";
      var pick0 = $("browserPick");
      if (pick0) { pick0.disabled = true; }
      var empty = ce("span", "dir-empty", OUT.error || "正在读取目录…");
      dirs.appendChild(empty);
      return;
    }
    var info = OUT.info;
    var current = info.exists ? info.path : info.requested;
    path.textContent = current + (info.exists ? "" : "（将新建）");
    path.title = current || "";
    var up = $("browserUp");
    var pick = $("browserPick");
    up.disabled = !info.parent || info.parent === info.path;
    if (pick) { pick.disabled = false; }
    var list = info.dirs || [];
    if (!list.length) {
      dirs.appendChild(ce("span", "dir-empty", "没有子目录"));
    } else {
      each(list, function (name) {
        var btn = ce("button", "dir-item", name);
        btn.type = "button";
        btn.title = joinPath(info.path, name);
        btn.onclick = function () { browseTo(joinPath(info.path, name)); };
        dirs.appendChild(btn);
      });
    }
    note.textContent = info.exists
      ? "点目录进入；底部按钮选择当前所在目录。"
      : "该目录还不存在；选它会让后端创建。";
  }

  function firstOutputMessage(list, extra) {
    var all = [];
    if (list) { each(list, function (m) { all.push(String(m)); }); }
    if (extra) { each(extra, function (m) { all.push(String(m)); }); }
    for (var i = 0; i < all.length; i++) {
      if (all[i].indexOf("输出目录") >= 0) { return all[i]; }
    }
    return "";
  }

  function renderOutStatus() {
    var host = $("outStatus");
    if (!host) { return; }
    var def = $("defaultOut");
    if (def) {
      def.textContent = defaultOutLabel();
      def.title = OPT.default_out || "out/studio_runs";
    }
    var text = "";
    var cls = "prov";
    var pfError = firstOutputMessage(PF && PF.errors, PF_LOCAL);
    var pfWarn = firstOutputMessage(PF && PF.warnings, null);

    if (!trim(S.outputDir)) {
      text = "默认：" + defaultOutLabel();
      host.title = OPT.default_out || "out/studio_runs";
    } else if (OUT.checking) {
      text = "正在检查目录…";
    } else if (OUT.error) {
      text = "✗ " + OUT.error;
      cls += " bad";
    } else if (pfError) {
      text = "✗ " + pfError;
      cls += " bad";
    } else if (OUT.localBad) {
      text = "✗ " + OUT.localText;
      cls += " bad";
    } else if (pfWarn) {
      text = "! " + pfWarn;
      cls += " warn";
    } else if (OUT.info) {
      if (OUT.info.writable === false) {
        text = "✗ 目录不可写：" + (OUT.info.requested || OUT.info.path);
        cls += " bad";
      } else if (!OUT.info.exists) {
        text = "✓ 目录可写 · 将新建此目录";
        cls += " ok";
      } else {
        text = "✓ 目录可写 · 已存在";
        cls += " ok";
      }
    } else {
      text = "写好后会检查目录。";
    }
    host.className = cls;
    host.textContent = text;
    updateGenEnabled();
  }

  /* ------------------------------------------------------------ 预检与预估 */

  function localEstimate() {
    var frames = clamp(intval(S.capture.frames, 20), 1, 20000);
    var sec = 0;
    var mb = 0;
    each(S.capture.sensors, function (key) {
      var cost = (OPT.cost && OPT.cost.sec) || {};
      var size = (OPT.cost && OPT.cost.mb) || {};
      sec += has(cost, key) ? num(cost[key], 1) : 1;
      mb += has(size, key) ? num(size[key], 0.5) : 0.5;
    });
    return {
      frames: frames,
      sensors: S.capture.sensors.slice(0),
      seconds: Math.round(sec * frames * 10) / 10,
      megabytes: Math.round(mb * frames * 10) / 10,
      note: "本地估算：后端未响应时按内置单帧成本推算",
      local: true
    };
  }

  function fmtDuration(seconds) {
    var sec = Math.max(0, Math.round(num(seconds, 0)));
    if (sec < 60) { return "约 " + sec + " 秒"; }
    var m = Math.floor(sec / 60);
    var s = sec % 60;
    if (m < 60) { return "约 " + m + " 分" + (s ? " " + s + " 秒" : ""); }
    var h = Math.floor(m / 60);
    var mm = m % 60;
    return "约 " + h + " 小时" + (mm ? " " + mm + " 分" : "");
  }

  function fmtSize(megabytes) {
    var mb = num(megabytes, 0);
    if (mb < 1024) { return "约 " + Math.round(mb) + " MB"; }
    return "约 " + (mb / 1024).toFixed(1) + " GB";
  }

  function renderEstimate(estimate, local) {
    var time = $("estTime");
    var disk = $("estDisk");
    var meta = $("estMeta");
    if (!estimate) {
      time.textContent = "—";
      disk.textContent = "—";
      meta.textContent = "完成预检后显示。";
      meta.title = "";
      return;
    }
    time.textContent = fmtDuration(estimate.seconds);
    disk.textContent = fmtSize(estimate.megabytes);
    var sensors = estimate.sensors || S.capture.sensors;
    meta.textContent = (estimate.frames || S.capture.frames) + " 帧 × " +
      sensors.length + " 模态" + (local ? " · 本地估算" : " · 按帧成本，不含启动");
    meta.title = estimate.note || "";
  }

  function pfLine(kind, text) {
    var line = ce("div", "pf-line " + kind);
    line.appendChild(ce("span", "dot"));
    line.appendChild(ce("span", "", text));
    return line;
  }

  function renderPreflight() {
    var body = $("pfBody");
    var dot = $("pfDot");
    if (!body || !dot) { return; }
    clear(body);
    body.className = "pf-body";

    if (PF_STATE === "checking") {
      dot.className = "state-dot checking";
      body.appendChild(ce("p", "pf-idle", "正在校验配置…"));
      return;
    }
    if (PF_STATE === "offline") {
      dot.className = "state-dot bad";
      body.appendChild(pfLine("error", "无法连接本地服务。确认后端已启动后再试。"));
      var retry = ce("button", "link-btn", "重试预检");
      retry.type = "button";
      retry.onclick = function () { runPreflight(); };
      var wrap = ce("p", "pf-note");
      wrap.appendChild(retry);
      body.appendChild(wrap);
      return;
    }

    var errors = PF_LOCAL.slice(0);
    var warns = [];
    if (PF && PF.errors) { each(PF.errors, function (m) { errors.push(m); }); }
    if (PF && PF.warnings) { each(PF.warnings, function (m) { warns.push(m); }); }

    if (!errors.length && !warns.length) {
      dot.className = "state-dot ok";
      body.appendChild(pfLine("ok", "未发现问题，可以生成。"));
      body.appendChild(ce("p", "pf-note", "预检只做几何与成本检查，不渲染；改动会自动重新校验。"));
      return;
    }

    dot.className = errors.length ? "state-dot bad" : "state-dot warn";
    each(errors, function (message) { body.appendChild(pfLine("error", message)); });
    each(warns, function (message) { body.appendChild(pfLine("warn", message)); });
    if (!errors.length) {
      body.appendChild(ce("p", "pf-note", "有提示，但可以生成；确认后继续。"));
    } else {
      body.appendChild(ce("p", "pf-note", "修正后会自动重新校验；错误清空后生成按钮解锁。"));
    }
  }

  function currentErrorCount() {
    var n = PF_LOCAL.length;
    if (PF && PF.errors) { n += PF.errors.length; }
    if (PF_STATE === "offline" && !n) { n = 1; }
    return n;
  }

  function updateGenEnabled() {
    var gen = $("genBtn");
    var prev = $("prevBtn");
    var still = $("stillBtn");
    var cancel = $("cancelBtn");
    var hint = $("genHint");
    if (!gen) { return; }
    var running = !!(currentJob && (currentJob.state === "queued" || currentJob.state === "running"));
    var busy = SUBMITTING || running;
    var errors = currentErrorCount();
    var disabled = busy || PF_STATE === "checking" || PF_STATE === "offline" ||
      PF_STATE === "idle" || errors > 0;
    gen.disabled = disabled;
    if (prev) { prev.disabled = disabled; }
    if (still) { still.disabled = disabled; }
    if (cancel) { cancel.hidden = !running; }
    if (hint) {
      if (busy) { hint.textContent = "任务进行中；完成前不能重复提交。"; }
      else if (PF_STATE === "checking") { hint.textContent = "正在校验配置…"; }
      else if (PF_STATE === "offline") { hint.textContent = "后端未连接，无法校验。"; }
      else if (errors > 0) { hint.textContent = "先修正预检错误；改完会自动重新校验。"; }
      else if (PF && PF.ok) { hint.textContent = "预检通过，可以生成。"; }
      else { hint.textContent = "等待预检结果…"; }
    }
  }

  function schedulePreflight(immediate) {
    if (pfTimer) { clearTimeout(pfTimer); }
    if (PF_STATE !== "offline") { PF_STATE = "checking"; }
    renderPreflight();
    updateGenEnabled();
    if (immediate === true) {
      runPreflight();
      return;
    }
    pfTimer = setTimeout(runPreflight, 400);
  }

  function runPreflight() {
    if (pfTimer) { clearTimeout(pfTimer); pfTimer = null; }
    var seq = ++PF_SEQ;
    PF_LOCAL = localErrors();
    if (PF_LOCAL.length) {
      PF = null;
      PF_STATE = "error";
      renderPreflight();
      renderEstimate(localEstimate(), true);
      renderOutStatus();
      updateGenEnabled();
      return;
    }
    PF_STATE = "checking";
    renderPreflight();
    updateGenEnabled();
    ajax("POST", API + "/preflight", buildPayload(), function (err, data) {
      if (seq !== PF_SEQ) { return; }
      if (err) {
        PF = null;
        PF_LOCAL = [err.error || "预检失败"];
        PF_STATE = err.offline ? "offline" : "error";
        if (err.offline) { conn(false); }
        renderPreflight();
        renderEstimate(localEstimate(), true);
        renderOutStatus();
        updateGenEnabled();
        return;
      }
      PF = data || {};
      PF_LOCAL = [];
      PF_STATE = PF.level === "error" ? "error" : (PF.level === "warn" ? "warn" : "ok");
      renderPreflight();
      renderEstimate(PF.estimate || localEstimate(), !PF.estimate);
      renderOutStatus();
      updateGenEnabled();
    });
  }

  function touch() {
    var still = $("stillBox");
    if (still) { still.hidden = true; }
    renderSummaries();
    updateNameHint();
    schedulePreflight(false);
  }

  /* ------------------------------------------------------------ 提交与轮询 */

  function showJobError(message, label) {
    var job = $("job");
    if (job) { job.hidden = false; }
    if (activeKind === "still") {
      showStillError(message);
      activeKind = "";
      var still = $("stillBtn");
      if (still) { still.textContent = "预览一张"; }
    }
    setText("jobState", label || "提交失败");
    setText("jobPct", "—");
    var bar = $("jobBar");
    if (bar) { bar.style.width = "0%"; }
    var wrap = $("jobOutWrap");
    if (wrap) { wrap.hidden = true; }
    setText("jobLog", String(message));
    var link = $("jobLink");
    if (link) { link.hidden = true; }
  }

  function renderJob() {
    var job = $("job");
    if (!job) { return; }
    if (!currentJob) { job.hidden = true; return; }
    job.hidden = false;

    var state = currentJob.state || "queued";
    var total = currentJob.total || 0;
    var done = currentJob.done || 0;
    var pct = total > 0 ? Math.min(100, Math.round(done * 100 / total))
      : (state === "done" ? 100 : 0);
    var label = JOB_LABEL[state] || state;
    if ((state === "running" || state === "queued") && total > 0) {
      label += " · " + done + " / " + total + " 帧";
    }
    setText("jobState", label);
    setText("jobPct", pct + "%");
    var bar = $("jobBar");
    if (bar) { bar.style.width = pct + "%"; }

    var outWrap = $("jobOutWrap");
    var out = $("jobOut");
    if (outWrap && out) {
      if (currentJob.out) {
        outWrap.hidden = false;
        out.textContent = currentJob.out;
        out.title = currentJob.out;
      } else {
        outWrap.hidden = true;
        out.textContent = "";
      }
    }

    var log = $("jobLog");
    if (log) {
      if (state === "failed") {
        var lines = [];
        if (currentJob.error_line) { lines.push(currentJob.error_line); }
        if (currentJob.error_tail) { lines = lines.concat(currentJob.error_tail); }
        if (!lines.length && currentJob.error) { lines.push(currentJob.error); }
        log.textContent = lines.join("\n") || "生成失败，后端没有给出更多信息。";
      } else if (state === "cancelled") {
        log.textContent = "已取消；已写入的输出目录可能留下半成品。";
      } else {
        var tail = currentJob.log || [];
        var last = "";
        for (var i = tail.length - 1; i >= 0; i--) {
          if (trim(tail[i])) { last = tail[i]; break; }
        }
        log.textContent = last || (state === "done" ? "完成。" : "等待后端输出…");
      }
    }

    var link = $("jobLink");
    if (link) {
      link.hidden = state !== "done";
      if (state === "done" && currentJob.id) {
        link.href = API + "/dataset/" + encodeURIComponent(currentJob.id);
      }
    }
    var still = $("stillBtn");
    if (still) {
      if (activeKind === "still" && (state === "queued" || state === "running")) {
        still.textContent = "预览一张 · " + pct + "%";
      } else {
        still.textContent = "预览一张";
      }
    }
    updateGenEnabled();
  }

  function pollJob() {
    if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
    if (!activeJobId) { return; }
    var id = activeJobId;
    ajax("GET", API + "/job/" + encodeURIComponent(id), null, function (err, data) {
      if (err) {
        /* 取消会杀掉子进程，紧接着的那一次轮询偶尔会拿到 501/连接重置；
           已取消的任务不该被一次瞬时错误改写成“失败”。 */
        if (currentJob && (currentJob.state === "cancelled" ||
            currentJob.state === "done" || currentJob.state === "failed")) {
          return;
        }
        pollFails++;
        if (pollFails < 4) {
          pollTimer = setTimeout(pollJob, 1500);
          return;
        }
        showJobError(err.error || "无法读取任务状态", "任务异常");
        return;
      }
      pollFails = 0;
      currentJob = data || {};
      renderJob();
      var state = currentJob.state || "";
      if (state === "queued" || state === "running") {
        pollTimer = setTimeout(pollJob, 1100);
      } else {
        finishJob();
      }
    });
  }

  function hideStillBox() {
    var box = $("stillBox");
    if (box) { box.hidden = true; }
  }

  function showStillPending() {
    var box = $("stillBox");
    var img = $("stillImg");
    var msg = $("stillMsg");
    if (!box || !img || !msg) { return; }
    box.hidden = false;
    img.hidden = true;
    img.removeAttribute("src");
    msg.className = "still-msg";
    msg.textContent = "正在渲染第 1 帧…";
  }

  function showStillImage(id) {
    var box = $("stillBox");
    var img = $("stillImg");
    var msg = $("stillMsg");
    if (!box || !img || !msg) { return; }
    box.hidden = false;
    img.hidden = false;
    msg.className = "still-msg";
    msg.textContent = "";
    img.onerror = function () {
      img.hidden = true;
      msg.className = "still-msg bad";
      msg.textContent = "预览图读取失败：/api/img/" + id + "/images/preview/000000.png";
    };
    img.onclick = function () { openStillPreview(id); };
    img.title = "点击查看大图";
    img.src = API + "/img/" + encodeURIComponent(id) + "/images/preview/000000.png";
  }

  function openStillPreview(id) {
    openModal("第 1 帧预览");
    var body = $("modalBody");
    var url = API + "/img/" + encodeURIComponent(id) + "/images/preview/000000.png";
    var img = ce("img", "still-large");
    img.src = url;
    img.alt = "第 1 帧预览";
    body.appendChild(img);
    body.appendChild(ce("p", "dataset-note",
      "这是第 1 帧的预览，正式生成会产出全部帧。"));
    var links = ce("div", "dataset-links");
    var raw = ce("a", "link", "打开原图");
    raw.href = url;
    raw.target = "_blank";
    raw.rel = "noopener";
    links.appendChild(raw);
    body.appendChild(links);
  }

  function showStillError(message) {
    var box = $("stillBox");
    var img = $("stillImg");
    var msg = $("stillMsg");
    if (!box || !img || !msg) { return; }
    box.hidden = false;
    img.hidden = true;
    msg.className = "still-msg bad";
    msg.textContent = String(message || "预览失败");
  }

  function finishJob() {
    updateGenEnabled();
    loadRuns();
    if (!currentJob) { return; }
    var wasKind = activeKind;
    if (wasKind === "still") {
      if (currentJob.state === "done") {
        showStillImage(currentJob.id);
      } else if (currentJob.state === "failed") {
        var reason = currentJob.error_line || currentJob.error || "预览失败";
        showStillError(reason);
      } else if (currentJob.state === "cancelled") {
        showStillError("预览已取消");
      }
      activeKind = "";
      var still = $("stillBtn");
      if (still) { still.textContent = "预览一张"; }
    }
    if (currentJob.state === "done") {
      if (wasKind === "still") {
        toast("第 1 帧预览已生成");
      } else {
        toast("生成完成" + (currentJob.out ? "：" + currentJob.out : ""));
      }
    } else if (currentJob.state === "failed") {
      toast(wasKind === "still" ? "预览失败，右栏有日志。" : "生成失败，右栏有日志。");
    } else if (currentJob.state === "cancelled") {
      toast("任务已取消。");
    }
  }

  function submit(kind) {
    if (SUBMITTING) { return; }
    if (currentJob && (currentJob.state === "queued" || currentJob.state === "running")) { return; }
    SUBMITTING = true;
    activeKind = kind;
    if (kind === "still") { showStillPending(); } else { hideStillBox(); }
    updateGenEnabled();
    var hint = $("genHint");
    if (hint) { hint.textContent = "正在提交任务…"; }
    var path = kind === "still" ? "/still" : (kind === "preview" ? "/preview" : "/generate");
    var payload = buildPayload();
    /* 一张预览的产物就是拼接图；即使操作员关掉了正式输出的拼接预览，
       这一次也要让它写出来，否则右栏没有图可显示。 */
    if (kind === "still") { payload.recipe.output.preview = true; }
    ajax("POST", API + path, payload, function (err, data) {
      SUBMITTING = false;
      if (err) {
        activeKind = "";
        updateGenEnabled();
        toast(err.error || "提交失败");
        showJobError(err.error || "提交失败");
        if (kind === "still") { showStillError(err.error || "预览提交失败"); }
        return;
      }
      currentJob = data || {};
      activeJobId = currentJob.id || "";
      pollFails = 0;
      renderJob();
      toast(kind === "still" ? "预览一张已提交" :
        (kind === "preview" ? "预览 8 帧任务已提交" : "生成任务已提交"));
      pollJob();
    });
  }

  function cancelJob() {
    if (!activeJobId) { return; }
    ajax("POST", API + "/job/" + encodeURIComponent(activeJobId) + "/cancel", {},
      function (err, data) {
        if (err) { toast(err.error || "取消失败"); return; }
        currentJob = data || {};
        pollFails = 0;
        renderJob();
        toast("已请求取消。");
        if (currentJob.state === "cancelled") {
          if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
          finishJob();
        } else {
          pollJob();
        }
      });
  }

  function copyText(text) {
    if (!text) { return; }
    function fallback() {
      var area = ce("textarea", "");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      var ok = false;
      try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
      document.body.removeChild(area);
      if (!ok) {
        var code = $("jobOut");
        if (code && window.getSelection && document.createRange) {
          var range = document.createRange();
          range.selectNodeContents(code);
          var sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(range);
          toast("已选中路径，按 Ctrl+C 复制");
        } else {
          toast("复制失败，请手动选择路径");
        }
      } else {
        toast("已复制路径");
      }
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () {
        toast("已复制路径");
      }, fallback);
    } else {
      fallback();
    }
  }

  /* ------------------------------------------------------------ 轨迹事实 */

  function renderPlan() {
    var host = $("planOut");
    if (!host) { return; }
    clear(host);
    if (!planData) { return; }
    if (planData.error) {
      host.appendChild(ce("p", "plan-error", planData.error));
      return;
    }
    var clips = planData.clips || [];
    if (!clips.length) {
      host.appendChild(ce("p", "plan-summary", "没有可用的轨迹段。"));
      return;
    }
    var summary = planData.summary || {};
    var parts = [];
    parts.push(clips.length + " 段");
    parts.push("每段 " + (summary.frames_per_clip || clips[0].frames || "?") + " 帧");
    if (summary.px_abs) {
      parts.push("目标 " + summary.px_abs[0] + "–" + summary.px_abs[1] + " px");
    }
    host.appendChild(ce("p", "plan-summary", parts.join(" · ")));

    var table = ce("table", "plan-table");
    var head = ce("tr", "");
    each(["段", "模式", "天气", "距离 km", "目标 px"], function (label) {
      head.appendChild(ce("th", "", label));
    });
    table.appendChild(head);
    each(clips.slice(0, 5), function (clip) {
      var row = ce("tr", "");
      var range = clip.range_m || [0, 0];
      var px = (clip.view_px && clip.view_px.seeker) || clip.target_px || ["—", "—"];
      row.appendChild(ce("td", "", clip.clip));
      row.appendChild(ce("td", "", clip.mode || "—"));
      row.appendChild(ce("td", "", weatherLabel(clip.weather)));
      row.appendChild(ce("td", "", (num(range[0], 0) / 1000).toFixed(1) + "–" +
        (num(range[1], 0) / 1000).toFixed(1)));
      row.appendChild(ce("td", "", px[0] + "–" + px[1]));
      table.appendChild(row);
    });
    host.appendChild(table);
  }

  function doPlan() {
    if (planBusy) { return; }
    if (S.traj.mode === "drawn" && !S.traj.smoothed) {
      planData = { error: "先点「优化轨迹」；生成和计划都只使用平滑后的手绘曲线。" };
      renderPlan();
      return;
    }
    planBusy = true;
    var btn = $("planBtn");
    if (btn) { btn.disabled = true; btn.textContent = "计算中…"; }
    ajax("POST", API + "/plan", buildPayload(), function (err, data) {
      planBusy = false;
      if (btn) { btn.disabled = false; btn.textContent = "预览轨迹事实"; }
      if (err) {
        planData = { error: err.error || "轨迹计算失败" };
      } else if (data && data.error) {
        planData = { error: data.error };
      } else {
        planData = data || {};
      }
      renderPlan();
    });
  }

  /* ------------------------------------------------------------ 运行历史 */

  function loadRuns() {
    var jobs = null;
    var datasets = null;
    var left = 2;
    function done() {
      left--;
      if (left > 0) { return; }
      RUNS = mergeRuns(jobs || [], datasets || []);
      renderRuns();
    }
    ajax("GET", API + "/jobs", null, function (err, data) {
      jobs = err ? [] : (data.jobs || []);
      done();
    });
    ajax("GET", API + "/datasets", null, function (err, data) {
      datasets = err ? [] : (data.datasets || []);
      done();
    });
  }

  function mergeRuns(jobs, datasets) {
    var map = {};
    var out = [];
    var key;
    each(jobs, function (job) {
      if (!job || !job.id) { return; }
      if (!map[job.id]) { map[job.id] = { id: job.id }; }
      map[job.id].job = job;
    });
    each(datasets, function (ds) {
      if (!ds || !ds.id) { return; }
      if (!map[ds.id]) { map[ds.id] = { id: ds.id }; }
      map[ds.id].ds = ds;
    });
    for (key in map) {
      if (has(map, key)) { out.push(map[key]); }
    }
    out.sort(function (a, b) {
      if (a.id < b.id) { return 1; }
      if (a.id > b.id) { return -1; }
      return 0;
    });
    return out;
  }

  function runState(run) {
    if (run.job) { return run.job.state || "queued"; }
    if (run.ds && run.ds.ok) { return "done"; }
    return "failed";
  }

  function renderRuns() {
    var host = $("runList");
    if (!host) { return; }
    clear(host);
    setText("runCount", String(RUNS.length));
    if (!RUNS.length) {
      host.appendChild(ce("p", "run-empty", "还没有运行记录。完成一次生成后会出现在这里。"));
      return;
    }
    each(RUNS.slice(0, 40), function (run) {
      var state = runState(run);
      var row = ce("div", "run-row");
      var dotClass = state === "running" || state === "queued" ? "running"
        : (state === "failed" ? "failed" : (state === "cancelled" ? "" : "done"));
      var dot = ce("span", "run-dot " + dotClass);
      row.appendChild(dot);

      var main = ce("div", "run-main");
      var name = ce("div", "run-name", run.id);
      if (run.job && run.job.out) { name.title = "输出：" + run.job.out; }
      main.appendChild(name);
      var meta = [];
      if (run.id.indexOf("still_") === 0) { meta.push("单帧预览"); }
      else if (run.id.indexOf("preview_") === 0) { meta.push("预览"); }
      if (run.ds && run.ds.frames !== null && run.ds.frames !== undefined) {
        meta.push(run.ds.frames + " 帧");
      }
      if (run.ds && run.ds.modalities && run.ds.modalities.length) {
        meta.push(run.ds.modalities.length + " 模态");
      }
      if (run.ds && run.ds.size_mb !== null && run.ds.size_mb !== undefined) {
        meta.push(run.ds.size_mb + " MB");
      }
      if (run.job) {
        meta.push(JOB_LABEL[run.job.state] || run.job.state);
        if (run.job.total) { meta.push((run.job.done || 0) + "/" + run.job.total + " 帧"); }
        if (run.job.seconds !== null && run.job.seconds !== undefined) {
          meta.push(run.job.seconds + " s");
        }
      }
      main.appendChild(ce("div", "run-meta", meta.join(" · ")));
      row.appendChild(main);

      var actions = ce("div", "run-actions");
      if (run.ds || state === "done") {
        var view = ce("a", "link", "查看");
        view.href = "#";
        view.onclick = function (ev) {
          if (ev) { ev.preventDefault(); }
          openDataset(run.id);
        };
        actions.appendChild(view);
      }
      if (run.ds) {
        var raw = ce("a", "link", "JSON");
        raw.href = API + "/dataset/" + encodeURIComponent(run.id);
        raw.target = "_blank";
        raw.rel = "noopener";
        actions.appendChild(raw);
      }
      row.appendChild(actions);
      host.appendChild(row);
    });
  }

  function toggleDrawer() {
    var drawer = $("drawer");
    var bar = $("drawerBar");
    var open = drawer.className.indexOf("open") < 0;
    drawer.className = "drawer" + (open ? " open" : "");
    bar.setAttribute("aria-expanded", open ? "true" : "false");
    if (open) { loadRuns(); }
  }

  /* ------------------------------------------------------------ 弹窗与产物 */

  function openModal(title) {
    var modal = $("modal");
    if (!modal) { return; }
    modal.hidden = false;
    setText("modalTitle", title || "");
    clear($("modalBody"));
  }

  function closeModal() {
    var modal = $("modal");
    if (modal) { modal.hidden = true; }
  }

  function openDataset(id) {
    openModal("产物");
    var body = $("modalBody");
    body.appendChild(ce("p", "pf-idle", "正在读取产物：" + id));
    ajax("GET", API + "/dataset/" + encodeURIComponent(id), null, function (err, data) {
      if (err) {
        clear(body);
        body.appendChild(ce("p", "plan-error", err.error || "无法读取产物"));
        var raw = ce("a", "link", "打开原始 JSON");
        raw.href = API + "/dataset/" + encodeURIComponent(id);
        raw.target = "_blank";
        raw.rel = "noopener";
        body.appendChild(raw);
        return;
      }
      setText("modalTitle", "产物 · " + id);
      renderDataset(data || {});
    });
  }

  function renderDataset(info) {
    var body = $("modalBody");
    clear(body);
    var stats = info.stats || {};
    var defs = [
      ["帧数", stats.frames], ["可见光", stats.optical_images],
      ["红外", stats.ir_images], ["雷达", stats.radar_frames],
      ["激光", stats.lidar_images], ["SAR", stats.sar_images]
    ];
    var grid = ce("div", "dataset-stats");
    var shown = 0;
    each(defs, function (item) {
      if (item[1] === null || item[1] === undefined || item[1] === "") { return; }
      if (item[0] !== "帧数" && !(num(item[1], 0) > 0)) { return; }
      var stat = ce("div", "stat");
      stat.appendChild(ce("b", "", item[1]));
      stat.appendChild(ce("span", "", item[0]));
      grid.appendChild(stat);
      shown++;
    });
    if (shown) {
      body.appendChild(grid);
    } else {
      body.appendChild(ce("p", "dataset-note",
        "这次运行还没有写出 stats.json：可能仍在生成，或生成失败。"));
    }
    if (info.dir) {
      var dirNote = ce("p", "dataset-note");
      dirNote.appendChild(ce("span", "", "输出目录："));
      dirNote.appendChild(ce("code", "", info.dir));
      body.appendChild(dirNote);
    }

    var cards = info.cards || [];
    if (cards.length) {
      var hosted = ce("div", "cards");
      each(cards.slice(0, 24), function (card) {
        var fig = ce("figure", "card");
        var link = ce("a", "");
        link.href = card.url;
        link.target = "_blank";
        link.rel = "noopener";
        var img = ce("img", "");
        img.src = card.url;
        img.alt = card.label || "";
        img.loading = "lazy";
        link.appendChild(img);
        fig.appendChild(link);
        var cap = ce("figcaption", "");
        cap.appendChild(ce("span", "", (card.label || card.mod || "") +
          (card.frame !== undefined && card.frame !== null ? " #" + card.frame : "")));
        var meta = [];
        if (card.w && card.h) { meta.push(card.w + "×" + card.h); }
        if (card.target_px) {
          if (isArray(card.target_px)) { meta.push(card.target_px[0] + " px"); }
          else { meta.push(card.target_px + " px"); }
        }
        cap.appendChild(ce("span", "c-meta", meta.join(" · ")));
        fig.appendChild(cap);
        hosted.appendChild(fig);
      });
      body.appendChild(hosted);
    } else {
      body.appendChild(ce("p", "dataset-note", "还没有可展示的样张。"));
    }

    var links = ce("div", "dataset-links");
    var raw = ce("a", "link", "打开原始 JSON");
    raw.href = API + "/dataset/" + encodeURIComponent(info.id);
    raw.target = "_blank";
    raw.rel = "noopener";
    links.appendChild(raw);
    if (info.variants && indexOf(info.variants, "interfered") >= 0) {
      var alt = ce("a", "link", "受扰镜像");
      alt.href = API + "/dataset/" + encodeURIComponent(info.id) + "?variant=interfered";
      alt.target = "_blank";
      alt.rel = "noopener";
      links.appendChild(alt);
    }
    body.appendChild(links);
  }

  /* ------------------------------------------------------------ 配方 */

  function openRecipes(focusSave) {
    openModal("配方库");
    var body = $("modalBody");
    /* 配方列表只在保存成功后刷新过，所以第一次打开永远是空的（磁盘上有配方
       却显示"还没有保存过配方"）。打开时先向服务端要一次。 */
    refreshRecipes();

    var saveSection = ce("div", "section");
    saveSection.appendChild(ce("h3", "", "保存当前配置"));
    var row = ce("div", "recipe-save");
    var input = ce("input", "");
    input.type = "text";
    input.id = "recipeSaveName";
    input.maxLength = 64;
    input.spellcheck = false;
    input.value = asciiName(S.recipeSource || recipeNameValue());
    row.appendChild(input);
    var save = ce("button", "btn primary small", "保存配方");
    save.type = "button";
    save.onclick = saveRecipeFromModal;
    row.appendChild(save);
    saveSection.appendChild(row);
    saveSection.appendChild(ce("p", "prov",
      "这里填配方文件名（ASCII：字母、数字、下划线、点、连字符）；配方内部会保留当前数据集名（允许中文）。"));
    body.appendChild(saveSection);

    var loadSection = ce("div", "section");
    loadSection.appendChild(ce("h3", "", "载入配方"));
    var list = ce("div", "recipe-list");
    list.id = "recipeList";
    loadSection.appendChild(list);
    loadSection.appendChild(ce("p", "prov",
      "载入会覆盖六个箱的当前取值，不会删除已有产物。"));
    body.appendChild(loadSection);
    renderRecipeList();
    if (focusSave) {
      setTimeout(function () { input.focus(); input.select(); }, 30);
    }
  }

  function renderRecipeList() {
    var host = $("recipeList");
    if (!host) { return; }
    clear(host);
    if (!RECIPES.length) {
      host.appendChild(ce("p", "empty", "还没有保存过配方。先配好一组参数，再在上方保存。"));
      return;
    }
    each(RECIPES, function (recipe) {
      var row = ce("div", "recipe-row");
      var main = ce("div", "");
      main.appendChild(ce("div", "r-title", recipe.title || recipe.name));
      var meta = [recipe.name];
      if (recipe.saved_at) { meta.push(recipe.saved_at); }
      main.appendChild(ce("div", "r-meta", meta.join(" · ")));
      row.appendChild(main);
      var load = ce("button", "btn quiet small", "载入");
      load.type = "button";
      load.onclick = function () { loadRecipeByName(recipe.name); };
      row.appendChild(load);
      host.appendChild(row);
    });
  }

  function refreshRecipes(cb) {
    ajax("GET", API + "/recipes", null, function (err, data) {
      if (!err) {
        RECIPES = (data && data.recipes) || [];
        OPT.recipes = RECIPES;
      }
      if (cb) { cb(err); }
    });
  }

  function saveRecipeFromModal() {
    var input = $("recipeSaveName");
    if (!input) { return; }
    var name = trim(input.value);
    if (!/^[A-Za-z0-9_.-]{1,64}$/.test(name)) {
      toast("配方名只能包含字母、数字、下划线、点和连字符。");
      input.focus();
      return;
    }
    var recipe = buildRecipe();
    recipe.title = S.name;
    ajax("POST", API + "/recipe/save", { name: name, recipe: recipe },
      function (err, data) {
        if (err) { toast(err.error || "保存失败"); return; }
        S.recipeSource = recipe.name;
        updateNameHint();
        toast("已保存配方 " + ((data && data.name) || name));
        refreshRecipes(function () { openRecipes(false); });
      });
  }

  function loadRecipeByName(name) {
    ajax("POST", API + "/recipe/load", { name: name }, function (err, data) {
      if (err) { toast(err.error || "载入失败"); return; }
      var recipe = (data && data.recipe) || {};
      applyRecipe(recipe);
      closeModal();
      toast("已载入配方：" + (recipe.title || recipe.name || name));
      schedulePreflight(true);
    });
  }

  function normalizeEntity(item) {
    if (!item || !item.kind) { return null; }
    var kind = String(item.kind);
    var defaults = ecmDefaults(kind);
    var params = {};
    var extra = {};
    var key;
    for (key in item) {
      if (!has(item, key) || key === "kind" || key === "group") { continue; }
      if (has(defaults, key)) { params[key] = item[key]; }
      else if (indexOf(COMMON, key) >= 0) { extra[key] = item[key]; }
    }
    if (!extra.id) { extra.id = "ecm_" + kind; }
    return { kind: kind, params: params, extra: extra };
  }

  function applyRecipe(recipe) {
    if (!recipe || typeof recipe !== "object") { return; }
    hideStillBox();
    if (recipe.title || recipe.name) {
      S.name = String(recipe.title || recipe.name);
    }
    S.recipeSource = recipe.name ? String(recipe.name) : "";

    var env = recipe.environment || {};
    if (labelIn(OPT.env_kinds, env.kind, "")) { S.env.kind = env.kind; }
    if (labelIn(OPT.weather, env.weather, "")) { S.env.weather = env.weather; }
    var bg = env.background || {};
    if (bg.mode === "hdri") { S.bg.mode = "hdri"; }
    else if (bg.mode === "upload") { S.bg.mode = "upload"; }
    else { S.bg.mode = "none"; }
    S.bg.path = bg.path || "";
    S.bg.dataUrl = "";

    var objects = recipe.objects || [];
    if (isArray(objects) && objects.length) {
      var obj = objects[0] || {};
      S.obj.asset = obj.asset || S.obj.asset;
      S.obj.cls = obj.cls || S.obj.cls;
      S.obj.scale = num(obj.scale, 1);
      S.obj.count = clamp(intval(obj.count, 1), 1, 64);
      S.size = sizeForScale(S.obj.scale);
      var asset = findAsset(S.obj.asset);
      if (asset) { S.obj.category = asset.category; }
    }

    var ecm = recipe.ecm || {};
    S.ecm.entities = [];
    if (isArray(ecm.entities)) {
      each(ecm.entities, function (item) {
        var entity = normalizeEntity(item);
        if (entity) { S.ecm.entities.push(entity); }
      });
    }
    if (!S.ecm.entities.length && ecm.mode && ecm.mode !== "none") {
      if (ecm.mode === "active" || ecm.mode === "mixed") {
        S.ecm.entities.push(newEntity("noise_jammer"));
      }
      if (ecm.mode === "passive" || ecm.mode === "mixed") {
        S.ecm.entities.push(newEntity("chaff_cloud"));
      }
    }

    var view = recipe.view || {};
    if (num(view.range_km, 0) > 0) { S.view.range_km = num(view.range_km, 8); }
    if (view.aspect === "head_on" || view.aspect === "pursuit" ||
        view.aspect === "look_down") { S.view.aspect = view.aspect; }
    if (num(view.fov_deg, 0) > 0) { S.view.fov_deg = num(view.fov_deg, 2.2); }
    S.view.views = view.views === "dual" ? "dual" : "single";

    S.traj.mode = "preset";
    S.traj.points = [];
    S.traj.smoothed = null;
    S.traj.stats = null;
    S.traj.warnings = [];
    S.traj.error = "";
    S.traj.alt_m = num(view.alt_m, 2500);
    S.traj.speed_mps = num(view.trajectory && view.trajectory.speed_mps, 250);
    S.traj.closed = false;
    var traj = view.trajectory || {};
    if (traj.mode === "drawn" && isArray(traj.points) && traj.points.length >= 2) {
      var smooth = [];
      var flat = [];
      each(traj.points, function (point) {
        if (!isArray(point) || point.length < 2) { return; }
        var x = num(point[0], 0);
        var y = num(point[1], 0);
        var z = point.length > 2 ? num(point[2], 2500) : 2500;
        smooth.push([x, y, z]);
        flat.push([x, y]);
      });
      if (smooth.length >= 2) {
        S.traj.mode = "drawn";
        S.traj.smoothed = smooth;
        S.traj.points = flat;
        S.traj.alt_m = smooth[0][2];
        S.traj.closed = !!traj.closed;
      }
    }

    var capture = recipe.capture || {};
    if (isArray(capture.sensors)) {
      var known = [];
      each(OPT.sensors, function (item) { known.push(item.key); });
      var sensors = [];
      each(capture.sensors, function (key) {
        if (indexOf(known, key) >= 0) { sensors.push(key); }
      });
      if (sensors.length) { S.capture.sensors = sensors; }
    }
    if (num(capture.frames, 0) > 0) { S.capture.frames = clamp(intval(capture.frames, 20), 1, 20000); }
    if (num(capture.fps, 0) > 0) { S.capture.fps = num(capture.fps, 10); }
    if (capture.seed !== null && capture.seed !== undefined) {
      S.capture.seed = clamp(intval(capture.seed, 7), 0, 4294967295);
    }

    var output = recipe.output || {};
    if (has(output, "preview")) { S.output.preview = !!output.preview; }
    if (has(output, "coco")) { S.output.coco = !!output.coco; }
    if (has(output, "yolo")) { S.output.yolo = !!output.yolo; }
    if (has(output, "readme")) { S.output.readme = !!output.readme; }

    renderAll();
    renderOutStatus();
  }

  /* ------------------------------------------------------------ 静态绑定 */

  function bindBoxes() {
    var boxes = document.querySelectorAll(".box");
    each(boxes, function (box) {
      var head = box.querySelector(".box-head");
      if (!head) { return; }
      head.onclick = function () {
        var open = box.className.indexOf("open") >= 0;
        each(document.querySelectorAll(".box"), function (other) {
          other.className = other === box && !open ? "box open" : "box";
          var h = other.querySelector(".box-head");
          if (h) { h.setAttribute("aria-expanded", other.className.indexOf("open") >= 0 ? "true" : "false"); }
        });
      };
      head.onkeydown = function (ev) {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          head.click();
        }
      };
    });
  }

  function syncSizeSeg() {
    var host = $("sizeSeg");
    if (!host) { return; }
    for (var i = 0; i < host.childNodes.length; i++) {
      var node = host.childNodes[i];
      if (node.nodeType !== 1) { continue; }
      node.className = node.getAttribute("data-v") === S.size ? "on" : "";
    }
  }

  function bindStatic() {
    var dataName = $("dataName");
    dataName.oninput = function () {
      S.name = dataName.value;
      S.recipeSource = "";
      updateNameHint();
      schedulePreflight(false);
    };

    var seed = $("seed");
    seed.oninput = function () {
      var v = parseInt(seed.value, 10);
      if (isFinite(v)) {
        S.capture.seed = clamp(v, 0, 4294967295);
        touch();
      }
    };
    seed.onchange = function () { seed.value = S.capture.seed; };

    var frames = $("frames");
    frames.oninput = function () {
      var v = parseInt(frames.value, 10);
      if (isFinite(v)) {
        S.capture.frames = clamp(v, 1, 20000);
        renderSummaries();
        touch();
      }
    };
    frames.onchange = function () { frames.value = S.capture.frames; };

    var fps = $("fps");
    fps.oninput = function () {
      var v = num(fps.value, S.capture.fps);
      S.capture.fps = clamp(v, 0.5, 120);
      touch();
    };
    fps.onchange = function () { fps.value = S.capture.fps; };

    var objCount = $("objCount");
    objCount.oninput = function () {
      var v = parseInt(objCount.value, 10);
      if (isFinite(v)) {
        S.obj.count = clamp(v, 1, 64);
        renderSummaries();
        touch();
      }
    };
    objCount.onchange = function () { objCount.value = S.obj.count; };

    var objScale = $("objScale");
    objScale.oninput = function () {
      var v = num(objScale.value, S.obj.scale);
      S.obj.scale = clamp(v, 0.02, 50);
      S.size = sizeForScale(S.obj.scale);
      syncSizeSeg();
      refreshAssetMeta();
      renderSummaries();
      touch();
    };
    objScale.onchange = function () { objScale.value = S.obj.scale; };

    var objCls = $("objCls");
    objCls.oninput = function () {
      S.obj.cls = objCls.value;
      touch();
    };

    var objPath = $("objPath");
    objPath.oninput = function () {
      S.obj.asset = objPath.value;
      refreshAssetMeta();
      touch();
    };
    objPath.onchange = function () {
      var asset = findAsset(trim(S.obj.asset));
      if (asset) {
        S.obj.category = asset.category;
        S.obj.cls = asset.cls;
        renderTarget();
        touch();
      }
    };

    var bgPath = $("bgPath");
    bgPath.oninput = function () {
      S.bg.path = bgPath.value;
      touch();
    };

    var bgFile = $("bgFile");
    bgFile.onchange = function () {
      var file = bgFile.files && bgFile.files[0];
      if (!file) { return; }
      var reader = new FileReader();
      reader.onload = function () {
        S.bg.dataUrl = String(reader.result || "");
        var thumb = $("bgThumb");
        if (thumb) { thumb.style.backgroundImage = 'url("' + S.bg.dataUrl + '")'; }
        toast("已读取本地预览；生成时请填写服务器可读路径。");
      };
      reader.readAsDataURL(file);
    };

    var slider = $("rangeSlider");
    slider.oninput = function () {
      S.view.range_km = clamp(num(slider.value, 8), 0.1, 100);
      $("rangeNum").value = S.view.range_km;
      refreshAssetMeta();
      renderSummaries();
      touch();
    };

    var rangeNum = $("rangeNum");
    rangeNum.oninput = function () {
      var v = num(rangeNum.value, S.view.range_km);
      S.view.range_km = clamp(v, 0.1, 100);
      slider.value = S.view.range_km;
      refreshAssetMeta();
      renderSummaries();
      touch();
    };
    rangeNum.onchange = function () { rangeNum.value = S.view.range_km; };

    var fovDeg = $("fovDeg");
    fovDeg.oninput = function () {
      var v = num(fovDeg.value, S.view.fov_deg);
      S.view.fov_deg = clamp(v, 0.1, 120);
      renderView();
      renderSummaries();
      refreshAssetMeta();
      touch();
    };
    fovDeg.onchange = function () { fovDeg.value = S.view.fov_deg; };

    var outputDir = $("outputDir");
    outputDir.oninput = function () {
      S.outputDir = outputDir.value;
      OUT.info = null;
      OUT.checking = false;
      OUT.error = "";
      OUT.localBad = false;
      OUT.localText = "";
      renderOutStatus();
      scheduleBrowse();
      touch();
    };
    outputDir.onchange = function () {
      if (trim(S.outputDir)) { browseTo(S.outputDir); }
    };

    var browseBtn = $("browseBtn");
    browseBtn.onclick = function () {
      OUT.open = !OUT.open;
      browseBtn.textContent = OUT.open ? "收起" : "浏览…";
      renderBrowser();
      if (OUT.open) {
        browseTo(trim(S.outputDir) || OPT.default_out || "");
      }
    };

    $("browserUp").onclick = function () {
      if (OUT.info && OUT.info.parent) { browseTo(OUT.info.parent); }
    };
    $("browserPick").onclick = function () {
      if (!OUT.info) { return; }
      var chosen = OUT.info.exists ? OUT.info.path : OUT.info.requested;
      if (!chosen) { return; }
      setOutputDir(chosen, true);
      OUT.open = false;
      browseBtn.textContent = "浏览…";
      renderBrowser();
      toast("输出目录：" + chosen);
      touch();
    };

    $("genBtn").onclick = function () { submit("generate"); };
    $("prevBtn").onclick = function () { submit("preview"); };
    $("stillBtn").onclick = function () { submit("still"); };
    $("cancelBtn").onclick = cancelJob;
    $("planBtn").onclick = doPlan;

    var trajCanvas = $("trajCanvas");
    trajCanvas.onclick = addTrajPointFromEvent;
    trajCanvas.oncontextmenu = function (ev) {
      ev.preventDefault();
      undoTrajPoint();
    };
    $("trajUndo").onclick = undoTrajPoint;
    $("trajClear").onclick = clearTrajectory;
    $("trajOptimize").onclick = optimizeTrajectory;
    $("trajClosed").onclick = function () {
      S.traj.closed = !S.traj.closed;
      invalidateTrajectory();
      renderTrajState();
      touch();
    };
    var trajAlt = $("trajAlt");
    trajAlt.oninput = function () {
      var v = num(trajAlt.value, S.traj.alt_m);
      S.traj.alt_m = clamp(v, 10, 30000);
      invalidateTrajectory();
      renderTrajState();
      touch();
    };
    trajAlt.onchange = function () { trajAlt.value = S.traj.alt_m; };
    var trajSpeed = $("trajSpeed");
    trajSpeed.oninput = function () {
      var v = num(trajSpeed.value, S.traj.speed_mps);
      S.traj.speed_mps = clamp(v, 1, 1000);
      invalidateTrajectory();
      renderTrajState();
      touch();
    };
    trajSpeed.onchange = function () { trajSpeed.value = S.traj.speed_mps; };
    $("copyOutBtn").onclick = function () {
      copyText(currentJob && currentJob.out ? currentJob.out : "");
    };
    $("drawerBar").onclick = toggleDrawer;
    $("recipeLoadBtn").onclick = function () { openRecipes(false); };
    $("recipeSaveBtn").onclick = function () { openRecipes(true); };
    $("modalClose").onclick = closeModal;
    $("modalBackdrop").onclick = closeModal;
    $("jobLink").onclick = function (ev) {
      if (currentJob && currentJob.id) {
        ev.preventDefault();
        openDataset(currentJob.id);
      }
    };
    document.onkeydown = function (ev) {
      if (ev.key === "Escape") {
        if ($("modal").hidden === false) { closeModal(); }
      }
    };
  }

  /* ------------------------------------------------------------ 启动 */

  function init() {
    bindBoxes();
    bindStatic();
    renderAll();
    renderPreflight();
    renderEstimate(null, false);
    renderOutStatus();
    loadPresets();
    loadAssets();
    loadBackdrops();
    var pc = $("placeCanvas");
    if (pc) { pc.addEventListener("click", placeClick); }
    var pb = $("placePreviewBtn");
    if (pb) { pb.addEventListener("click", placePreview); }
    var sp = $("placeSpeed");
    if (sp && !sp.dataset.bound) {
      sp.dataset.bound = "1";
      sp.value = String(S.place.speed || 80);
      sp.addEventListener("change", function () {
        S.place.speed = num(sp.value, 80);
        renderPlace();
        touch();
      });
    }
    var cb = $("placeClearBtn");
    if (cb) {
      cb.addEventListener("click", function () {
        S.place.points = [];
        S.place.url = "";
        S.place.placed = false;
        renderPlace();
      });
    }
    loadRuns();
    refreshRecipes();
  }

  init();
})();
