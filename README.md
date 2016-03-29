[![Build Status](https://travis-ci.com/tingelst/game.svg?token=JjgE1aL6Uy3yn3YFR5Nx&branch=master)](https://travis-ci.com/tingelst/game)

# **G**eometric **A**lgebra **M**ultivector **E**stimation

GAME is framework for estimation of multivectors in geometric algebra with focus on, but not limited to, the Euclidean and conformal model. 

## Installation

The main external dependency of GAME is the Ceres optimization framework from Google. 

On OSX we recommend building the homebrew formula from source:
``` bash
brew tap homebrew/science
brew install ceres-solver --build-from-source
```

On Linux we recommend following the installation instructions found here: http://ceres-solver.org/building.html.

To build GAME follow the steps below:
``` bash
$ git clone --recursive https://github.com/tingelst/game.git
$ cd game
$ mkdir build && cd build 
$ cmake .. && make
```
