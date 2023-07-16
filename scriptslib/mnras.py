import matplotlib.pyplot as plt

FIG_WIDTH = 8.27
FONT_SIZE = 14

MPL_STYLE = "default"


def size_from_aspect(aspect: float, scale: float = 1.0) -> tuple[float, float]:
    """
    aspect = height / width
    """
    return FIG_WIDTH * scale, aspect * FIG_WIDTH * scale


def set_style():
    plt.style.use(MPL_STYLE)
