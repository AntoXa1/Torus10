#!/bin/bash          

echo Torus10 config amd compile script 

#make all clean

./configure --with-coord=cylindrical \
 --with-problem=torus10 --with-gas=mhd --with-flux=hlld \
--with-order=2p --enable-mpi --with-integrator=vl \
--enable-fofc 

# --with-order=2p --enable-mpi --with-integrator=vl --enable-fft \

# make all MACHINE=macosx

# make all MACHINE=macosxmpi

#make all MACHINE=atorusmpi


# ./configure --with-coord=cylindrical \
#  --with-problem=torus10 --with-gas=mhd --with-flux=hlld \
# --with-order=2p --enable-mpi --with-integrator=vl \
# --enable-fofc --enable-fft



