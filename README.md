# Scripts for the paper about black holes in the Milky Way

### `softening_distance.py`
**Goal**: determine the optimal softening distance that sould be applied when evolving the system of
a given number of points.

### `bh_orbits.py`
**Goal**: determine realistic orbits for the black holes to fall onto the centre of the potantial.
Potential is computed as a multipole approximation of a set of particles representing the galaxy.

### `bh_orbits_visualizer.py`
**Goal**: predict possible orbits for the black hole from the satellite galaxy. 

### `system_generator.py`
**Goal**: create $N$-body approximation for the Milky Way as a merger of two galaxies.

### `eccentricity_example.py`
**Goal**: create example of the orbits of the black hole in a given potential. 


### To be updated.


# Development

## Guidelines

### Making plotting time reasonable

**Motivation**: all of the scripts usually model some system and then plot some graph. 
If the system is not trivial, the computation part can take a lot of time: up to several hours.
In order to plot the data fast, one should save it to file and then read it with visualiser.

This also gives an ability to change the visualisation without recomputing everything.

**General guidelines**:

- All scripts which take a lot of time to complete should be split into two parts: computing and visualising ones.
- Computing part should (obviously) compute the thing and save the result into the folder `{script_name}/results`
- This result should be the smallest physical representation of the stuff that was computed. 

**For example**:
- If the computer evolves system of N points and visualiser then shows the density of the points in a given point, computing part of the script *should* save the density map or (depending on a particular case) the list of points. It *should not* save the RGB map.
- If the computing part computes some (physical) value on a grid of points, it *should* save this value as a 2D array and then plot it with visualiser part as needed. It *should not* save the contour lines or any normalisation of this data.

### Making scripts parallelisable

**Motivation**: not all of the scripts are able to be parallelised by their design.
This means that if one launches the process it would utilise only one core out of many that any modern computer usually has. 
To make the computations more effective, one should try to use `concurrent` module from Python as much as possible. 

**General guidelines**:

- If the computation part of the script can be split in separate independent computation processes without the significant harm to the perfomance, it should do so.
- Parameters of each process should be stored in dataclass and instantiated as a global list of instances of this dataclass. 
- The business logic of the program should be located in `process` function of the script which accepts at least one argument: single instance of the `Params` class form the previous point.
- Inside the `if __name__ == "__main__"` section one should use 
    ```python
    with futures.ProcessPoolExecutor(max_workers=9) as executor:
            executor.map(process, params)
    ```
    to launch the computations of the independent parts as a separate processes on a separate cores.
- This approach comes with a number of tradeoffs:
    - If one computation is dependent on the results of the other, this approach is not applicable and the computations shpuld be consecutive.
    - One cannot easily transfer data and variables between processes (though this is possible using sockets but it is *significantly* harder (and slower) than usual).
    - Each computation part should be primarily CPU bound (i.e. mostly computations) and not IO bound (i.e. mostly writing to/reading from the disk).
    - One should not using global mutable variables to avoid data race.

**For example**:
- If the computation part of the script is the N-body model of some given system with variable initial velocity, mass, etc. one should create
    ```python
    @dataclass
    class Params:
        mass: float

    parameters = [Params(10), Params(20), Params(30)]
    ```
    and move the logic of the integration into function 
    ```python
    def process(param: Params):
        ...
    ```
    and then use `concurrent.futures` to parallelise calls to `process` sunction for each member of `parameters` list.
