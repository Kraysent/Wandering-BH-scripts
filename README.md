# Scripts for the paper about black holes in the Milky Way

### `models_resolution`

The command will run all of the models with different resolutions and show two figures with distance and bound mass plots.
You might need host and satellite files in `models_resolution/models/sat.csv` and `models_resolution/models/sat.csv`.

```shell
python3 main.py models-resolution
```

For flags description use 

```shell
python3 main.py models-resolution --help
```

### `models_example`

This command will run the actual model with two galaxies and show the pictures of this process.

```shell
python3 main.py models-example
```

For flags description use 

```shell
python3 main.py models-example --help
```
