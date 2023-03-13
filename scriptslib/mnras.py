FIG_WIDTH = 8.27
FONT_SIZE = 14


def size_from_aspect(aspect: float) -> tuple[float, float]:
    """
    aspect = height / width
    """
    return FIG_WIDTH, aspect * FIG_WIDTH
