import click

import all_models.main as module_all_models
import models_example.main as example
import models_resolution.main as resolution
import models_velocity_vector.main as velocities
from bh_orbits import eccentricity_example


class CommonCommand(click.core.Command):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.insert(
            0,
            click.core.Option(("--mode",), help="Mode for the objects: [paper, presentation]", default="paper"),
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
def models_resolution(save, save_trajectories, cached, mode, **kwargs):
    if not cached:
        resolution.model(save_trajectories, save)
    else:
        resolution.load(save, mode)


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
@click.option(
    "-sp",
    "--separate-plot",
    is_flag=True,
    type=bool,
    help="Already have rgb maps saved in results/bins? This option will plot them in a separate planes each. Does not ignore -s option.",
)
def models_example(save, plot, separate_plot, **kwargs):
    if plot:
        example.plot_plane(save)
    elif separate_plot:
        example.plot_separate_pic(save)
    else:
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
def models_velocities(plot, save, mode, **kwargs):
    if plot:
        velocities.plot(save, mode)
    else:
        velocities.compute()


@cli.command(cls=CommonCommand, name="bh-orbits-example")
def bh_orbits_example(**kwargs):
    eccentricity_example.model()


@cli.command(cls=CommonCommand)
def all_models(**kwargs):
    module_all_models.model()


if __name__ == "__main__":
    cli()
