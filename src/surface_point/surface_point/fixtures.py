"""Deterministic stress fixtures; all geometry is boxes, not mesh surfaces."""
from observed_obb import Box


def scenes():
    back = Box((0,0,.10),(.10,.16,.20))
    return {
        'clear': (back,),
        'side_occlusion': (back,Box((.2,.07,.10),(.08,.10,.20))),
        'split': (back,Box((.2,0,.10),(.08,.04,.20))),
        'short_occluder_U': (back,Box((.2,0,.04),(.08,.06,.08))),
        'hole_center': (back,Box((.2,0,.10),(.08,.06,.08))),
        'rotated': (Box((0,0,.10),(.10,.16,.20),.55),Box((.2,.07,.10),(.08,.10,.20))),
        'fully_hidden': (back,Box((.2,0,.10),(.08,.20,.22))),
    }
