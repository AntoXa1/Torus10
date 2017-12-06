#!/bin/bash          

echo Torus10 config script 

setenv MPI_VERBOSE 
setenv MPI_DISPLAY_SETTINGS 

./configure --with-coord=cylindrical \
 --with-problem=torus9 --with-gas=mhd --with-flux=hlld \
--with-order=2p --enable-mpi --with-integrator=vl \
--enable-fofc

# module load comp-intel/2018.0.128

