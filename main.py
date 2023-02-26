import click
from models_resolution.main import model, load


@click.group()
def cli():
    pass


@cli.command()
@click.option("-s", "--save", is_flag=True, type=bool, help="Save to PDF file or just show figures?")
@click.option("-st", "--save-trajectories", is_flag=True, type=bool, help="Save trajectories to separate file?")
@click.option("-c", "--cached", is_flag=True, type=bool, help="Already have trajectories on the files? This option will plot them. Ignores -st option but does not ignore -s one.")
def models_resolution(save, save_trajectories, cached):
    if not cached:
        model(save_trajectories, save)
    else:
        load(save)


if __name__ == "__main__":
    cli()
