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
