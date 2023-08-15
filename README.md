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


# Development

## Guidelines

### Making plotting time reasonable

**Motivation**: all of the scripts usually model some system and then plot some graph. 
If the system is not trivial, the computation part can take a lot of time: up to several hours.
In order to plot the data fast, one should save it to file and then read it with visualiser.

This also gives an ability to change the visualisation without recomputing everything.

**General guidelines are**:

- All scripts which take a lot of time to complete should be split into two parts: computing and visualising ones.
- Computing part should (obviously) compute the thing and save the result into the folder `{script_name}/results`
- This result should be the smallest physical representation of the stuff that was computed. 

**For example**:
- If the computer evolves system of N points and visualiser then shows the density of the points in a given point, computing part of the script *should* save the density map or (depending on a particular case) the list of points. It *should not* save the RGB map.
- If the computing part computes some (physical) value on a grid of points, it *should* save this value as a 2D array and then plot it with visualiser part as needed. It *should not* save the contour lines or any normalisation of this data.
