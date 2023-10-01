from matplotlib.axes import Axes
from matplotlib.patches import Patch
import numpy as np
import scriptslib
from scriptslib import particles as sparticles, plot as splot
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from amuse.lab import units
import matplotlib.pyplot as plt

from scriptslib import mnras
from scriptslib import physics

RESULTS_DIR = "example_models/{}"
DENSITY_CUTOFF_RADIUS = 20


def prepare_axes(ax, time: float):
    title = ax.set_title(f"{time:.02f} Gyr", y=0.97, pad=-14, fontsize=mnras.FONT_SIZE)
    title.set_bbox(dict(edgecolor="black", boxstyle="round,pad=0.3", facecolor="white"))
    ax.set_box_aspect(1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])


def show():
    # using ndarray to ensure that this is matrix and not list of lists
    times = np.array(
        [
            [0.0, 0.5],
            [1.5, 2.0],
            [3.0, 3.5],
        ]
    )
    space_fig, space_axes = plt.subplots(*times.shape, sharex="all", sharey="all")
    space_fig.set_size_inches(mnras.size_from_aspect(3 / 2))
    density_fig, density_ax = plt.subplots()
    density_ax_inset: Axes = inset_axes(density_ax, width="60%", height="80%", loc="upper right", borderpad=1.2)

    for ix, iy in np.ndindex(times.shape):
        prepare_axes(space_axes[ix, iy], times[ix, iy])

    space_axes[0, 0].legend(
        handles=[
            Patch(facecolor=splot.colors[0], edgecolor=splot.colors[0], label="Host"),
            Patch(facecolor=splot.colors[3], edgecolor=splot.colors[3], label="Satellite"),
        ],
        prop={"size": mnras.FONT_SIZE},
        loc="lower right",
    )

    scalebar = AnchoredSizeBar(
        space_axes[0, 0].transData,
        10,
        "10 kpc",
        "lower right",
        pad=0.4,
        color="black",
        frameon=False,
        size_vertical=0.5,
        fontproperties=fm.FontProperties(size=mnras.FONT_SIZE),
    )

    space_axes[0, 1].add_artist(scalebar)

    axis_extent = DENSITY_CUTOFF_RADIUS
    densities_history = []

    for ix, iy in np.ndindex(times.shape):
        particles = sparticles.pipe(
            scriptslib.read_hdf5(f"system_generator/models/{times[ix,iy]:.03f}.hdf5"),
            sparticles.select(lambda is_barion: is_barion, ["is_barion"]),
        )
        host = particles.select(lambda system: system == "host", ["system"])
        sat = particles.select(lambda system: system == "sat", ["system"])

        host_positions = host.position.value_in(units.kpc)
        sat_positions = sat.position.value_in(units.kpc)

        extent = [-axis_extent, axis_extent, -axis_extent, axis_extent]
        rgb_map = splot.plot_colored_points(
            xs=[host_positions[:, 0], sat_positions[:, 0]],
            ys=[host_positions[:, 1], sat_positions[:, 1]],
            colors=[splot.colors[0], splot.colors[3]],
            extent=extent,
            threshold=2,
        )
        rgb_map = np.stack(rgb_map, axis=2)
        params = dict(extent=extent, interpolation="nearest", aspect="auto")
        space_axes[ix, iy].imshow(rgb_map, **params)
        space_axes[ix, iy].set_xlim(-axis_extent, axis_extent)
        space_axes[ix, iy].set_ylim(-axis_extent, axis_extent)

        radii, densities = physics.get_density_distribution(
            particles,
            [0, 0, 0] | units.kpc,
            cutoff_radius=DENSITY_CUTOFF_RADIUS | units.kpc,
        )
        radii = radii.value_in(units.kpc)
        densities = densities.value_in(units.MSun / units.kpc**3)
        densities_history.append(densities)

        density_ax.plot(
            radii,
            densities_history[0] - densities,
            label=f"{times[ix, iy]:.03f} Gyr",
        )
        density_ax_inset.plot(
            radii,
            densities,
            label=f"{times[ix, iy]:.03f} Gyr",
        )

    density_ax_inset.set_xlabel("$r$, kpc", fontsize=mnras.FONT_SIZE)
    density_ax_inset.set_ylabel("density, $M_{\odot}$ kpc$^{-3}$", fontsize=mnras.FONT_SIZE)
    density_ax_inset.set_xlim(0, DENSITY_CUTOFF_RADIUS)
    density_ax_inset.set_yscale("log")
    density_ax_inset.grid(True)
    density_ax_inset.legend(prop={"size": mnras.FONT_SIZE})
    density_ax.set_xlabel("$r$, kpc", fontsize=mnras.FONT_SIZE)
    density_ax.set_ylabel("density residuals, $M_{\odot}$ kpc$^{-3}$", fontsize=mnras.FONT_SIZE)
    density_ax.set_xlim(0, DENSITY_CUTOFF_RADIUS)
    density_ax.grid(True)
    density_fig.tight_layout()
    density_fig.savefig(RESULTS_DIR.format("density_example.pdf"))

    space_fig.tight_layout()
    space_fig.subplots_adjust(wspace=0, hspace=0)
    space_fig.savefig(RESULTS_DIR.format("merger_example.pdf"))


if __name__ == "__main__":
    show()
