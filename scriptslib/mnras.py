FIG_WIDTH = 8.27
FONT_SIZE = 14


def size_from_aspect(aspect: float, scale: float = 1.) -> tuple[float, float]:
    """
    aspect = height / width
    """
    return FIG_WIDTH * scale, aspect * FIG_WIDTH * scale

