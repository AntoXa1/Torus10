#PBS -S /bin/csh

##!/bin/bash          

echo Torus10 config amd compile script 

setenv MPI_VERBOSE 
setenv MPI_DISPLAY_SETTINGS 

module load comp-intel/2018.0.128 mpi-sgi/mpt

./configure --with-coord=cylindrical \
 --with-problem=torus9 --with-gas=mhd --with-flux=hlld \
--with-order=2p --enable-mpi --with-integrator=vl \
--enable-fofc

make all MACHINE=atorus_super_mpi



