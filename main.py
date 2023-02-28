import click
import models_resolution.main as resolution
import models_example.main as example
import matplotlib.pyplot as plt


class CommonCommand(click.core.Command):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.insert(0, click.core.Option(("--style",), help="Style of matplotlib graphs", default="default"))


@click.group()
def cli():
    pass


@cli.command(cls=CommonCommand)
@click.option("-s", "--save", is_flag=True, type=bool, help="Save to PDF file or just show figures?")
@click.option("-st", "--save-trajectories", is_flag=True, type=bool, help="Save trajectories to separate file?")
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


@click.option("-s", "--save", is_flag=True, type=bool, help="Save to PDF file or just show figures?")
@cli.command(cls=CommonCommand)
def models_example(save, style):
    plt.style.use(style)
    example.model(save)


if __name__ == "__main__":
    cli()
