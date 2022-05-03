# RPCF DNS Solver

## Build Solver
To build the solver is simple. Navigate to the solver directory and run the following lines.

```
make clean
make
```

The result will be an executable called `rpcf` in the solver directory.If there are any compile errors (most likely from certain header files not being in
the include path) then you're on your own lmao :joy:

For ease it's a good idea to add a symlink to the appropriate directory in your path, e.g.

```
ln -s /absolute/path/to/rpcf /absolute/path/to/symlink/directory/
```

Now it is possible, given a directory containing the correct files for initialise a run, to use the rpcf in that folder (or provide a path to it) to run a
simulation.

## Install Post-Processing
To install the pre-processing code is even simpler. This code is written in Python and is distributed with a setup.py script. Therefore, simply run the
command

```
pip install -e /absolute/path/to/directory/with/setup.py
```

and all the packages will be importable from the environment it was installed (ideally a virtual environment).
