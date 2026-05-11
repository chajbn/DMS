# DMS 接口文档

> 版本: 1.0  
> 用途: 为下游消费者提供人脸关键点、头姿、视线方向、嘴巴闭合度的数据接口说明

---

## 1. 整体调用流程

```
FaceMeshDetector.detect(frame)
        │
        ├──→ HeadPoseEstimator.estimate(landmarks_px, w, h, transform)  →  head_pose
        ├──→ GazeEstimator.estimate(face_data)                           →  gaze_result
        ├──→ MouthDetector.compute_mar(landmarks_px)                     →  mar
        │
        └──→ DistractionDetector.update(head_pose, gaze_result, mar)    →  state, details
```

---

## 2. FaceMeshDetector — 人脸关键点

### 2.1 接口

```python
detector = FaceMeshDetector(max_num_faces=1, min_detection_confidence=0.5)
face_data = detector.detect(image_bgr)       # image_bgr: BGR uint8 numpy array (H,W,3)
```

### 2.2 输出 `face_data` (dict)

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `landmarks_pixel` | `np.ndarray[int32]` | `(478, 2)` 或 `(0, 2)` | 478 个关键点的像素坐标 `[x, y]`。无人脸时为空数组 |
| `landmarks_3d` | `np.ndarray[float32]` | `(478, 3)` 或 `(0, 3)` | 归一化 3D 坐标 `[x, y, z]`，范围约 [0,1] |
| `blendshapes` | `dict[str, float]` 或 `None` | 52 个键 | MediaPipe 52 个面部混合形状系数，值域 [0, 1] |
| `transform` | `np.ndarray` 或 `None` | `(4, 4)` | 面部变换矩阵（模型→相机），含旋转和平移 |

### 2.3 常用关键点索引

| 索引 | 部位 | 说明 |
|------|------|------|
| 1 | 鼻尖 | 头姿轴线原点 |
| 152 | 下巴 | 头姿估计 |
| 33 / 263 | 左眼外角 / 右眼外角 | 头姿估计、视线估计 |
| 133 / 362 | 左眼内角 / 右眼内角 | 视线估计 |
| 468 / 473 | 右虹膜中心 / 左虹膜中心 | 虹膜跟踪 |
| 61 / 291 | 左嘴角 / 右嘴角 | 嘴巴宽度基准 |
| 13 / 14 | 上唇内侧 / 下唇内侧 | 嘴巴开合度 (MAR) |
| 0 / 17 | 上唇顶部 / 下唇底部 | 嘴巴外轮廓 |

### 2.4 Blendshapes 列表 (52 项)

下游可按需取用，与本文档相关的核心项:

| 名称 | 含义 | 用途 |
|------|------|------|
| `eyeBlinkLeft` / `eyeBlinkRight` | 左/右眼闭合度 (0=睁眼, 1=闭眼) | 眨眼/疲劳 |
| `eyeLookUpLeft` / `eyeLookUpRight` | 左/右眼向上看 | 视线方向 |
| `eyeLookDownLeft` / `eyeLookDownRight` | 左/右眼向下看 | 视线方向 |
| `eyeLookInLeft` / `eyeLookOutRight` | 向左看 | 视线方向 |
| `eyeLookOutLeft` / `eyeLookInRight` | 向右看 | 视线方向 |
| `jawOpen` | 下颚张开度 | 嘴巴开合 |
| `mouthClose` | 嘴巴闭合度 | 嘴巴开合 |
| `mouthPucker` / `mouthSmile*` | 嘴型 | 表情 |

---

## 3. HeadPoseEstimator — 头姿估计

### 3.1 接口

```python
head_pose = HeadPoseEstimator()
pose = head_pose.estimate(landmarks_px, image_w, image_h, transform=None)
```

### 3.2 输出

`pose` 为 `(yaw, pitch, roll)` 元组，单位 **度 (°)**；检测失败时返回 `None`。

| 分量 | 正方向 | 负方向 | 典型范围 |
|------|--------|--------|----------|
| **yaw** (偏航) | 头向右转 `+` | 头向左转 `−` | ±90° |
| **pitch** (俯仰) | 抬头 `+` | 低头 `−` | ±60° |
| **roll** (翻滚) | 头向右倾 `+` | 头向左倾 `−` | ±45° |

### 3.3 坐标约定

```
模型坐标系 (右手系):           相机坐标系:
  +x → 右                       +x → 右
  +y → 上                       +y → 下
  +z → 后 (从面部指向脑后)       +z → 前 (指向场景)
```

---

## 4. GazeEstimator — 视线方向

### 4.1 接口

```python
gaze_est = GazeEstimator(ear_threshold=0.22)
gaze_result = gaze_est.estimate(face_data)
```

### 4.2 输出 `gaze_result` (dict)

| 字段 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `gaze_yaw` | `float` | [−45, +45] ° | 视线水平偏角: 正=看右, 负=看左 |
| `gaze_pitch` | `float` | [−30, +30] ° | 视线垂直偏角: 正=看下, 负=看上 |
| `avg_ear` | `float` | [0, ~0.35] | 平均眼睛纵横比 (EAR), 低值=闭眼 |
| `blink_left` | `float` | [0, 1] | 左眼闭合度 blendshape (0=睁, 1=闭) |
| `blink_right` | `float` | [0, 1] | 右眼闭合度 blendshape (0=睁, 1=闭) |

- 无人脸时 `estimate()` 返回 `None`
- 主方法: 通过 blendshape 系数换算角度；备选: 虹膜位置相对眼角偏移

### 4.3 EAR 阈值参考

| EAR 范围 | 状态 | 含义 |
|----------|------|------|
| ≥ 0.22 | 睁眼 | 正常 |
| 0.15 ~ 0.22 | 眯眼 | 轻度疲劳 |
| < 0.15 | 闭眼 | 严重疲劳/眨眼 |

---

## 5. MouthDetector — 嘴巴闭合度

### 5.1 接口

```python
mouth_det = MouthDetector(mar_yawn_threshold=0.65, mar_open_threshold=0.35)
mar = mouth_det.compute_mar(landmarks_px)
```

### 5.2 输出

| 字段 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `mar` | `float` | [0, ~2.0+] | Mouth Aspect Ratio，无人脸时返回 `None` |

### 5.3 MAR 计算公式

```
MAR = ||lip_upper_inner(13)  − lip_lower_inner(14)||
      ───────────────────────────────────────────────
      ||mouth_left(61) − mouth_right(291)||
```

### 5.4 MAR 阈值参考

| MAR 范围 | 状态 | 含义 |
|----------|------|------|
| < 0.35 | **闭合** | 嘴巴关闭或微张 |
| 0.35 ~ 0.65 | **张开** | 说话、吃东西 |
| > 0.65 | **打哈欠** | 张大嘴，可作为疲劳辅助信号 |

---

## 6. DistractionDetector — 综合分心/疲劳判定

### 6.1 接口

```python
dms = DistractionDetector(
    yaw_threshold=25.0,         # 头姿偏航阈值 (°)
    pitch_threshold=20.0,       # 头姿俯仰阈值 (°)
    gaze_yaw_threshold=20.0,    # 视线偏航阈值 (°)
    gaze_pitch_threshold=15.0,  # 视线俯仰阈值 (°)
    ear_drowsy_threshold=0.20,  # 闭眼疲劳阈值
    mar_yawn_threshold=0.65,    # 哈欠阈值
    drowsy_duration=1.5,        # 持续闭眼判疲劳时长 (秒)
    distracted_duration=1.0,    # 持续分心判分心时长 (秒)
    yawn_duration=1.0,          # 持续张嘴判哈欠时长 (秒)
)

state, details = dms.update(head_pose, gaze_result, mar)
```

### 6.2 输出

#### `state` — DistractionState 枚举值

| 枚举值 | 字符串值 | 含义 |
|--------|----------|------|
| `NORMAL` | `"Normal"` | 正常驾驶 |
| `SLIGHT` | `"Slight Distraction"` | 轻度分心 |
| `DISTRACTED` | `"Distracted"` | 分心 (头+视线同时偏离) |
| `DROWSY` | `"Drowsy"` | 疲劳 (持续闭眼 或 持续打哈欠) |
| `NO_FACE` | `"No Face"` | 未检测到人脸 |

#### `details` — dict

| 字段 | 类型 | 出现条件 | 说明 |
|------|------|----------|------|
| `state` | `str` | 始终 | 当前状态字符串 |
| `yaw` | `float` | 有头姿 | 头部偏航角，保留1位小数 |
| `pitch` | `float` | 有头姿 | 头部俯仰角，保留1位小数 |
| `roll` | `float` | 有头姿 | 头部翻滚角，保留1位小数 |
| `gaze_yaw` | `float` | 有视线 | 视线偏航角，保留1位小数 |
| `gaze_pitch` | `float` | 有视线 | 视线俯仰角，保留1位小数 |
| `ear` | `float` | 有视线 | 平均眼睛纵横比，保留3位小数 |
| `mar` | `float` | 有嘴巴 | 嘴巴纵横比，保留3位小数 |
| `fps` | `float` | main.py 中加入 | 实时帧率 |

### 6.3 判定逻辑

```
疲劳判定优先级 (任一满足即判 DROWSY):
  1. avg_ear < 0.20 持续 ≥ 1.5s  →  闭眼疲劳
  2. mar    > 0.65 持续 ≥ 1.0s  →  哈欠疲劳

分心判定 (疲劳不满足时):
  头姿 yaw>25° 且 视线 yaw>20°    →  DISTRACTED (持续 ≥1.0s)
  头姿 或 视线 单一超标            →  SLIGHT     (持续 ≥1.5s)
  均未超标                        →  NORMAL
```

---

## 7. 下游集成示例

### 7.1 获取每帧全量数据的伪代码

```python
from face_mesh_detector import FaceMeshDetector
from head_pose_estimator import HeadPoseEstimator
from gaze_estimator import GazeEstimator
from mouth_detector import MouthDetector
from distraction_detector import DistractionDetector

detector   = FaceMeshDetector(max_num_faces=1)
head_pose  = HeadPoseEstimator()
gaze_est   = GazeEstimator()
mouth_det  = MouthDetector()
dms        = DistractionDetector()

# 每帧调用
face_data = detector.detect(frame)
landmarks = face_data['landmarks_pixel']
blendshapes = face_data['blendshapes']
transform   = face_data['transform']

pose   = head_pose.estimate(landmarks, frame_w, frame_h, transform)
gaze   = gaze_est.estimate(face_data)
mar    = mouth_det.compute_mar(landmarks)
state, details = dms.update(pose, gaze, mar)

# details 即为下游可直接消费的结构化数据
# 例: {"state": "Normal", "yaw": 3.2, "pitch": -1.5, "roll": 0.8,
#       "gaze_yaw": 5.1, "gaze_pitch": -2.3, "ear": 0.312, "mar": 0.128}
```

### 7.2 仅需关键点 (不经 DistractionDetector)

```python
# landmarks_pixel   →  shape (478, 2), 可直接取任意索引
# blendshapes       →  dict, 可直接取如 blendshapes['jawOpen']
# mar               →  float, MouthDetector 单独输出
```

---

## 8. 数据类型速查

| 模块 | 输出 | Python 类型 | 无人脸时 |
|------|------|-------------|----------|
| FaceMeshDetector | `landmarks_pixel` | `np.ndarray (N,2)` | `(0,2)` 空数组 |
| FaceMeshDetector | `blendshapes` | `dict[str,float]` | `None` |
| HeadPoseEstimator | head_pose | `tuple(y,p,r)` | `None` |
| GazeEstimator | gaze_result | `dict` | `None` |
| MouthDetector | mar | `float` | `None` |
| DistractionDetector | state | `DistractionState` | `NO_FACE` |
| DistractionDetector | details | `dict` | 仅含 `state` 字段 |
