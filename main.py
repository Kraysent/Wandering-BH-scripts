import click
from models_resolution.main import model, load


@click.group()
def cli():
    pass


@cli.command()
@click.option("-s", "--save", type=bool)
@click.option("-st", "--save-trajectories", type=bool)
@click.option("-c", "--cached", type=bool)
def models_resolution(save, save_trajectories, cached):
    if not cached:
        model(save_trajectories, save)
    else:
        load(save)


if __name__ == "__main__":
    cli()
