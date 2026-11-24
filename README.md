# pairwise-video-deinterlace

## 领域

游戏、图形与多媒体

## 主题

视频去隔行

## 说明

场序识别、运动梳齿与乱序输出。

初始状态下 `media/video_deinterlace.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。

## 修复点与实现映射

| 缺陷 | 修复函数 | 做法 |
| --- | --- | --- |
| 容器场序标记与实际扫描顺序相反 | `field_order` | 分别按 TFF/BFF 假设把场 bob 成全帧率序列，取帧间 SAD 最小者为真实场序；与声明不符时置 `flipped` 并返回 `corrected`，静态内容回退声明值 |
| 混合去隔行在快速横移上留梳齿 | `motion_comb` / `edge_weave` / `adaptive_blend` | 三行二阶差分检出运动掩码并膨胀；静态区保留 weave，运动区改用**场内**边缘方向插值（ELA，参考行只取同场间隔 2 行），不再混用对面场 |
| 奇数高度裁切使奇偶场错位一行 | `odd_crop` | 起始行与裁切高度强制偶数对齐，保证两场行数一致、相位不翻转 |
| 异步解码多线程乱序输出、偶场先到 | `decode_order` / `timestamp` | 按 PTS 稳定排序，统计逆序数，校验奇偶交替与时间戳网格 |
| 丢场 | `cadence` / `fallback` | 按 PTS 网格定位缺口、推断缺失场奇偶并计数；用前后真实场时间方向混合补齐，补场带 `synthetic` 标记 |
| 同批文件两遍帧数/内容不一致 | `batch_determinism` | 逐帧 SHA-256 哈希（含形状/dtype）比对，标出帧数、形状、像素三类差异 |

端到端管线 `deinterlace` 串联以上步骤，并在内部对自身跑两遍做哈希自检；
`metrics` 给出梳齿能量与相对 PSNR，`audit` 汇总缺陷告警，`profile` 剖析各阶段耗时。

## 约定

- 帧数组形状 `(F,H,W)` 或 `(F,H,W,C)`，隔行帧高度必须为偶数；
- 顶场（TFF）取偶数行、底场（BFF）取奇数行；
- 场记录为 `{"parity": "top"|"bottom", "pts": float, "data": ndarray}`；
- 各入口收到标量整数（如契约测试的 `1`）时进入内置合成隔行素材的自检路径，
  跑的是同一套真实逻辑而非占位返回；正常数组输入走完整实现。

`tests/test_fixes.py` 针对六条修复给出行为级测试（共 44 个），
`tests/test_video_deinterlace.py` 为原契约测试（15 个）。
