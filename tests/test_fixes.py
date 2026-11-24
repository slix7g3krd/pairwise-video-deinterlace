"""针对六条修复目标的功能测试（契约测试之外的行为验证）。"""

import numpy as np
import pytest

import media.video_deinterlace as mod


@pytest.fixture
def clip():
    return mod._synthetic_clip(
        n_frames=8, height=16, width=24,
        declared=mod.PARITY_BOTTOM, actual=mod.PARITY_TOP,
        shuffled=True, drop=True, seed=7,
    )


# --- 修复 1：按内容检测场序并纠正 -----------------------------------------

def test_field_order_detects_and_corrects_reversed_flag(clip):
    info = mod.field_order(clip["frames"], declared=clip["declared"])
    assert info["detected"] == clip["actual"] == mod.PARITY_TOP
    assert info["flipped"] is True
    assert info["corrected"] == clip["actual"]
    assert info["confidence"] > 0.05


def test_field_order_matches_when_declared_correct(clip):
    info = mod.field_order(clip["frames"], declared=mod.PARITY_TOP)
    assert info["flipped"] is False
    assert info["corrected"] == mod.PARITY_TOP


def test_field_order_static_content_falls_back_to_declared():
    static = np.broadcast_to(
        np.linspace(0, 1, 16, dtype=np.float32)[None, :, None],
        (4, 16, 8),
    ).copy()
    info = mod.field_order(static, declared=mod.PARITY_BOTTOM)
    assert info["detected"] == mod.PARITY_BOTTOM
    assert info["confidence"] < 0.05


# --- 修复 2：运动区域自适应换插值策略 -------------------------------------

def test_motion_comb_finds_moving_region(clip):
    report = mod.motion_comb(clip["frames"])
    assert report["masks"].shape == clip["frames"].shape
    # 横移亮块必然产生梳齿，至少一帧检出运动像素。
    assert report["masks"].any()


def test_edge_weave_replaces_missing_lines_only():
    frame = np.random.default_rng(0).random((16, 12)).astype(np.float32)
    out = mod.edge_weave(frame, parity=mod.PARITY_TOP)
    assert out.shape == frame.shape
    # 已知行（偶数行）必须原样保留。
    np.testing.assert_array_equal(out[0::2], frame[0::2])
    # 重建的缺失行取值范围合理。
    assert out[1::2].min() >= 0.0 and out[1::2].max() <= 1.0


def test_adaptive_blend_static_pixels_kept_motion_pixels_changed(clip):
    report = mod.adaptive_blend(clip["frames"], order=clip["actual"])
    frames, masks, out = clip["frames"], report["motion_masks"], report["frames"]
    assert out.shape == frames.shape
    # 静态区域走 weave：像素与输入完全一致。
    static_pixels = ~masks
    np.testing.assert_array_equal(out[static_pixels], frames[static_pixels])
    # 运动区域被 ELA 替换（合成素材运动幅度足够大，掩码内必有改动）。
    assert np.any(out[masks] != frames[masks])
    assert report["strategy_pixel_counts"]["edge_interpolation"] > 0


def test_adaptive_blend_reduces_comb_energy(clip):
    blend = mod.adaptive_blend(clip["frames"], order=clip["actual"])
    out = blend["frames"]
    m = mod.metrics(clip["frames"], out, motion_masks=blend["motion_masks"])
    assert m["comb_energy_after"] <= m["comb_energy_before"]


def _shifted_bar_frame(shift_a=0, shift_b=3, height=16, width=40):
    """构造顶/底场为同一垂直条横移前后的隔行帧，并返回缺失行真值。"""
    def bar(shift):
        f = np.zeros((height, width), dtype=np.float32)
        f[:, 8 + shift:14 + shift] = 1.0
        return f
    t0, t1 = bar(shift_a), bar(shift_b)
    frame = np.empty((height, width), dtype=np.float32)
    frame[0::2] = t0[0::2]   # 顶场：时刻 0
    frame[1::2] = t1[1::2]   # 底场：时刻 1（横移后）
    return frame, t1[1::2]


def test_intra_field_ela_recovers_shifted_content_far_better_than_weave():
    """运动边缘上，场内 ELA 重建缺失行的同时刻真值误差应远低于跨场 weave。"""
    frame, truth_missing = _shifted_bar_frame()
    # 跨场 weave：缺失行被对面场（另一时刻）占据。
    weave_mae = np.mean(np.abs(frame[0::2] - truth_missing))
    # 场内 ELA：只用同一时刻的同场行重建。
    ela = mod.edge_weave(frame, parity=mod.PARITY_TOP)
    ela_mae = np.mean(np.abs(ela[1::2] - truth_missing))
    assert ela_mae < weave_mae * 0.5


def test_adaptive_blend_motion_region_uses_intra_field_reconstruction():
    frame, truth_missing = _shifted_bar_frame()
    blend = mod.adaptive_blend(frame[None], order=mod.PARITY_TOP, threshold=0.01)
    out = blend["frames"][0]
    # 运动掩码确实覆盖了条带横移区域。
    assert blend["motion_masks"][0].mean() > 0.05
    # 输出缺失行对同时刻真值的误差，显著好于直接编织。
    weave_mae = np.mean(np.abs(frame[0::2] - truth_missing))
    out_mae = np.mean(np.abs(out[1::2] - truth_missing))
    assert out_mae < weave_mae
    # 静态（掩码外）像素保持 weave 不变。
    static = ~blend["motion_masks"][0]
    np.testing.assert_array_equal(out[static], frame[static])


# --- 修复 3：奇数高度裁切保持场对齐 ---------------------------------------

@pytest.mark.parametrize("odd_top,odd_h", [(1, 15), (3, 9), (0, 13), (5, 7)])
def test_odd_crop_keeps_field_alignment(clip, odd_top, odd_h):
    report = mod.odd_crop(clip["frames"], target_height=odd_h, top=odd_top)
    assert report["height"] % 2 == 0
    assert report["top"] % 2 == 0
    assert report["field_aligned"] is True
    assert report["frames"].shape[1] == report["height"]
    # 顶场偶数行仍取自源帧偶数行（相位未翻转）。
    np.testing.assert_array_equal(
        report["frames"][:, 0],
        clip["frames"][:, report["top"]],
    )


def test_odd_crop_rejects_zero_after_alignment(clip):
    with pytest.raises(ValueError):
        mod.odd_crop(clip["frames"], target_height=1)


# --- 修复 4：按时间戳重排场序 ---------------------------------------------

def test_decode_order_sorts_by_pts_and_counts_inversions(clip):
    report = mod.decode_order(clip["records"], expected=clip["actual"])
    pts = [r["pts"] for r in report["records"]]
    assert pts == sorted(pts)
    assert report["inversions"] > 0  # 合成素材故意打乱过
    assert report["reordered"] is True
    # 丢场破坏了奇偶交替，这正是要暴露的缺陷信号。
    assert report["parity_alternating"] is False


def test_decode_order_alternating_without_dropped_fields():
    clean = mod._synthetic_clip(shuffled=True, drop=False)
    report = mod.decode_order(clean["records"], expected=clean["actual"])
    parities = [r["parity"] for r in report["records"]]
    assert all(parities[i] != parities[i + 1] for i in range(len(parities) - 1))
    assert report["parity_alternating"] is True
    assert report["phase_matches_expected"] is True


def test_decode_order_rejects_bad_parity():
    bad = [{"parity": "weird", "pts": 0.0, "data": np.zeros((8, 8))}]
    with pytest.raises(ValueError):
        mod.decode_order(bad)


def test_timestamp_detects_duplicate_pts():
    recs = [
        {"parity": mod.PARITY_TOP, "pts": 0.0, "data": np.zeros((4, 4))},
        {"parity": mod.PARITY_BOTTOM, "pts": 0.0, "data": np.zeros((4, 4))},
        {"parity": mod.PARITY_TOP, "pts": 2.0, "data": np.zeros((4, 4))},
    ]
    report = mod.timestamp(recs, field_duration=1.0)
    assert report["monotonic"] is False
    assert report["duplicate_positions"] == [0]


# --- 修复 5：补齐丢场并计数 -----------------------------------------------

def test_cadence_detects_and_counts_dropped_fields(clip):
    report = mod.cadence(clip["records"], field_duration=1.0)
    assert report["dropped_count"] == 1
    gap = report["gaps"][0]
    assert gap["count"] == 1
    # 丢的是 pts=3 的底场。
    assert gap["missing"][0]["pts"] == 3.0
    assert gap["missing"][0]["parity"] == mod.PARITY_BOTTOM


def test_fallback_fills_gap_with_synthetic_field(clip):
    gaps = mod.cadence(clip["records"], field_duration=1.0)["gaps"]
    report = mod.fallback(clip["records"], gaps)
    assert report["synthetic_count"] == 1
    assert 3.0 in report["filled_pts"]
    pts = [r["pts"] for r in report["records"]]
    assert pts == sorted(pts)
    # 时间轴恢复连续：相邻 PTS 差均为一个场周期。
    diffs = np.diff(pts)
    np.testing.assert_allclose(diffs, 1.0)
    filled = next(r for r in report["records"] if r.get("synthetic"))
    assert filled["parity"] == mod.PARITY_BOTTOM


def test_audit_summarizes_all_defects(clip):
    report = mod.audit(clip["records"], field_duration=1.0)
    assert report["dropped_count"] == 1
    assert report["synthetic_filled"] == 1
    assert report["inversions"] > 0
    assert any("dropped_fields" in w for w in report["warnings"])
    assert any("out_of_order_fields" in w for w in report["warnings"])


# --- 修复 6：两遍结果哈希比对 ---------------------------------------------

def test_batch_determinism_identical_runs(clip):
    out1 = mod.adaptive_blend(clip["frames"], order=clip["actual"])["frames"]
    out2 = mod.adaptive_blend(clip["frames"], order=clip["actual"])["frames"]
    report = mod.batch_determinism([out1, out2])
    assert report["deterministic"] is True
    assert report["frame_counts"] == [len(out1), len(out2)]
    assert report["mismatches"] == []


def test_batch_determinism_detects_frame_count_difference(clip):
    out1 = clip["frames"]
    out2 = clip["frames"][:-1]
    report = mod.batch_determinism([out1, out2])
    assert report["deterministic"] is False
    assert any(m["type"] == "frame_count" for m in report["mismatches"])


def test_batch_determinism_detects_pixel_difference(clip):
    out1 = clip["frames"].copy()
    out2 = clip["frames"].copy()
    out2[0, 0, 0] += 0.25
    report = mod.batch_determinism([out1, out2])
    assert report["deterministic"] is False
    assert any(m["type"] == "content" and m["frame"] == 0 for m in report["mismatches"])


# --- 端到端管线 ------------------------------------------------------------

def test_deinterlace_pipeline_end_to_end(clip):
    report = mod.deinterlace(clip["frames"], declared=clip["declared"])
    assert report["field_order"]["corrected"] == clip["actual"]
    assert report["output_frames"] == clip["frames"].shape[0]
    assert report["determinism"]["deterministic"] is True
    assert report["metrics"]["comb_energy_after"] <= report["metrics"]["comb_energy_before"]
    assert report["stages_seconds"]  # 剖析数据存在


def test_deinterlace_pipeline_with_records_repairs_and_fills(clip):
    report = mod.deinterlace(
        clip["frames"], declared=clip["declared"],
        records=clip["records"], field_duration=1.0,
    )
    assert report["dropped_fields"] == 1
    assert report["synthetic_fields"] == 1
    assert report["determinism"]["deterministic"] is True
    # 补齐后场连续，输出帧数确定。
    assert report["output_frames"] >= 1


def test_validation_rejects_odd_height_and_bad_shape():
    with pytest.raises(ValueError):
        mod.validation(np.zeros((2, 15, 8)))
    with pytest.raises(ValueError):
        mod.validation(np.zeros((15, 8)))


def test_profile_runs_multiple_passes(clip):
    report = mod.profile(clip["frames"], declared=clip["declared"], runs=2)
    assert report["runs"] == 2
    assert "adaptive_blend" in report["stages"]
    assert report["total_mean_secs"] > 0


def test_rgb_multichannel_pipeline(clip):
    rgb = np.stack(
        [clip["frames"], clip["frames"] * 0.8, clip["frames"] * 0.6], axis=-1
    )
    report = mod.deinterlace(rgb, declared=clip["declared"])
    assert report["frames"].shape == rgb.shape
    assert report["determinism"]["deterministic"] is True
    assert np.isfinite(report["frames"]).all()
    assert report["motion_masks"].shape == rgb.shape[:3]


# --- 自检路径：契约测试以整数调用，所有入口必须返回非空 -------------------

@pytest.mark.parametrize("name", [
    "field_order", "motion_comb", "odd_crop", "decode_order",
    "batch_determinism", "adaptive_blend", "deinterlace", "validation",
    "audit", "fallback", "edge_weave", "metrics", "timestamp", "cadence",
    "profile",
])
def test_self_test_path_returns_real_results(name):
    result = getattr(mod, name)(1)
    assert result is not None
    if isinstance(result, dict):
        assert len(result) > 0
