import pytest
import media.video_deinterlace as mod

CASES = [
    ("field_order", (1,)),
    ("motion_comb", (1,)),
    ("odd_crop", (1,)),
    ("decode_order", (1,)),
    ("batch_determinism", (1,)),
    ("adaptive_blend", (1,)),
    ("deinterlace", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("edge_weave", (1,)),
    ("metrics", (1,)),
    ("timestamp", (1,)),
    ("cadence", (1,)),
    ("profile", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
