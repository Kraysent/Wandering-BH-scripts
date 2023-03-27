import click
import matplotlib.pyplot as plt

import bh_orbits.main as bh_orbit
import bh_orbits.nbody_system_gen as galaxy_generate
import dynamical_friction_example.main as friction_example
import models_example.main as example
import models_resolution.main as resolution
import models_velocity_vector.main as velocities


class CommonCommand(click.core.Command):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.insert(
            0,
            click.core.Option(("--style",), help="Style of matplotlib graphs", default="default"),
        )


@click.group()
def cli():
    pass


@cli.command(cls=CommonCommand)
@click.option(
    "-s",
    "--save",
    is_flag=True,
    type=bool,
    help="Save to PDF file or just show figures?",
)
@click.option(
    "-st",
    "--save-trajectories",
    is_flag=True,
    type=bool,
    help="Save trajectories to separate file?",
)
@click.option(
    "-c",
    "--cached",
    is_flag=True,
    type=bool,
    help="Already have trajectories on the files? This option will plot them. Ignores -st option but does not ignore -s one.",
)
def models_resolution(save, save_trajectories, cached, style):
    plt.style.use(style)
    if not cached:
        resolution.model(save_trajectories, save)
    else:
        resolution.load(save)


@cli.command(cls=CommonCommand)
@click.option(
    "-s",
    "--save",
    is_flag=True,
    type=bool,
    help="Save to PDF or just show figures?",
)
@click.option(
    "-p",
    "--plot",
    is_flag=True,
    type=bool,
    help="Already have rgb maps saved in results/bins? This option will plot them in a single plane. Does not ignore -s option.",
)
def models_example(save, plot, style):
    plt.style.use(style)
    example.model(save, plot)


@cli.command(cls=CommonCommand)
@click.option(
    "-p",
    "--plot",
    is_flag=True,
    type=bool,
    help="Show data that was generated when running without this flag.",
)
@click.option(
    "-s",
    "--save",
    is_flag=True,
    type=bool,
    help="Save to PDF or just show figures?",
)
def models_velocities(plot, save, style):
    plt.style.use(style)

    if plot:
        velocities.plot(save)
    else:
        velocities.compute()


@cli.command(cls=CommonCommand)
@click.option(
    "-g",
    "--generate",
    is_flag=True,
    type=bool,
    help="Generate N body system; basically, the new galaxy",
)
def bh_orbits(generate, style):
    plt.style.use(style)

    if generate:
        galaxy_generate.generate_snapshot()
    else:
        bh_orbit.compute()


@cli.command(cls=CommonCommand)
def df_example(style):
    plt.style.use(style)
    friction_example.model()


if __name__ == "__main__":
    cli()
