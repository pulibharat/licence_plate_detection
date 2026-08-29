def pad_box(x1, y1, x2, y2, frame_shape, pad_ratio=0.15):
    """Expand a box by pad_ratio on each side, clamped to frame bounds."""

    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)

    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(frame_shape[1], x2 + pad_x)
    py2 = min(frame_shape[0], y2 + pad_y)

    return px1, py1, px2, py2
