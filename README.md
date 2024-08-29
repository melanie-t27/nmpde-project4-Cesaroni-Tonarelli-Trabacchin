# Cardiac Electrophysiology
This repository contains the code of the project for the Numerical Methods for Partial Differential Equations course @Polimi (pfof. Alfio Maria Quarterooni, TA Michele Bucelli).
**Students**:
- [Sabrina Cesaroni](https://github.com/SabrinaCesaroni)
- [Melanie Tonarelli](https://github.com/melanie-t27)
- [Tommaso Trabacchin](https://github.com/tommasotrabacchinpolimi) 

## Introduction
This project solves the Monodomain equation coupled with the Bueono-Orovio ionic model, employing the Finite Elements Method.
Various type of tissue can be simulated, including the epicardium, the myd-miocardium and the endocardium.

## Prerequisites
In order to run the software a few libraries are required:
- deal.II, built with MPI and TRilinos support
- MPI, for parallel execution
- gmsh, needed only for the mesh generation

## Mesh generation
The software needs a mesh as one of the inputs. Various formats are accepted, including the .msh one.
A script is provided in the scripts folder, which can be used to generate a 21 mm x 7 mm x 3 mm cuboid mesh, with an adjustable refinement.
In order to use it, the following instructions should be followed:
```bash

$ mkdir build
$ cd build
$ cmake ..
$ make GenerationMesh MESH_STEP=[mesh_step]
```
where `mesh_step` is the chosen mesh refinement (in meters).
The mesh will be created in the newly created `meshes` folder.
## Compiling
To build the executable, make sure you have loaded the needed modules with
```bash
$ module load gcc-glibc dealii
```
Then run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```
## Execution
The executable will be created into `build`, and can be executed through
```bash
$ mpirun -n 4 nmpde -fn ../meshes/cuboid_v2.msh -T 0.05 -dT 5e-3 -tfe 0.5 -tode 0.5 -ct 0 -tt 0 -o test
```
where
- `n` is the number of processes
- `fn` is the path to the mesh
- `T` the time (in seconds) 
- `-dT` is the time step (in seconds)
- `-tfe` is the theta value for the Monodomain equation
- `tode` is the theta value for the system of ODEs of the Ionic Model
- `-ct` is the coupler type, in particular
   - 0 corrsponds to the `ICI COupler`
   - 1 corrsponds to the `GI Coupler`
   - 2 corrsponds to the `SVI Coupler`
- `-tt` is the tissue type to simulate, in particular
   - 0 corresponds to the epicardium
   - 1 corresponds to the myd-miocardium
   - 2 corresponds to the enocardium
- `-o` is the prefix attached to all output files
