#!/Users/dora/Library/Enthought/Canopy_32bit/User/bin/python

import subprocess as subproc
import sys

#caseToDo = 'torus_hd'

caseToDo = 'torus_mhd'

#caseToDo = 'hkdisk_mhd'        

#caseToDo = 'hkdisk_hd'

#./configure --with-coord=cylindrical --with-problem=cylwindrotb


PATH_BASE='/Users/dora/WORK/ECLIPSE_SPACE/'
PATH_BASE = '/Users/dora/WORK/ECLIPSE_SPACE/AthenaWind'

import socket
name=socket.gethostname()

if name == 'atorus':
    PATH_BASE = '/local/data/atorus1/dora/PROJECTS'

if name == 'Antons-MacBook-Pro.local':
    PATH_BASE = "/Users/dora/WORK/ECLIPSE_SPACE"

PATH = PATH_BASE + '/AthenaWind/'

print("current Path: ",  PATH)


if caseToDo == '1':    
    problemToConfig = '--with-problem=cylwindrot'
    methodGasOrMHD =  '--with-gas=hydro'
    inputFile = '../tst/cylindrical/athinput.cylwindrot-3D'    

if caseToDo == '2':    
    problemToConfig = '--with-problem=cylwindrotb'
    methodGasOrMHD =  '--with-gas=mhd'
    inputFile = '../tst/cylindrical/athinput.cylwindrotb-2D'

if caseToDo == '3':    
    problemToConfig = '--with-problem=torus9'
    methodGasOrMHD =  '--with-gas=hydro'    
    inputFile = '../tst/cylindrical/athinput.torus9_hydro_2D'

if caseToDo == 'hkdisk_mhd':    
    problemToConfig = '--with-problem=hkdisk'
    methodGasOrMHD =  '--with-gas=mhd'
    inputFile = '../tst/cylindrical/athinput.hkdisk-3D'
    
if caseToDo == 'hkdisk_hd':
    problemToConfig = '--with-problem=hkdisk'
    methodGasOrMHD =  '--with-gas=hydro'
    inputFile = '../tst/cylindrical/athinput.hkdisk-3D'


if caseToDo == 'torus_hd':    
    problemToConfig = '--with-problem=torus9'
    methodGasOrMHD =  '--with-gas=hydro'    
    inputFileList = ['../tst/cylindrical/athinput.torus9_hydro_2D', '../tst/cylindrical/athinput.torus9_hydro_2D_2']
    inputFile = '../tst/cylindrical/athinput.torus9_hydro_2D'
    

if caseToDo == 'torus_mhd':    

    problemToConfig = '--with-problem=torus9'

    methodGasOrMHD =  '--with-gas=mhd'
    
    inputFile = '../tst/cylindrical/athinput.torus9_hydro_2D'
    

subproc.check_call(['rm', '-f', './bin/*.bin'])

 
if caseToDo == 'torus_hd':
    METHOD = '--with-flux=roe'
else:
   METHOD = '--with-flux=hlld'
   
    # METHOD = '--with-flux=hlle'
    
   ORDER = '--with-order=2p'
    
#   ORDER = '--with-order=3p'

   Integrator = '--with-integrator=vl'

compLev = '0123'

#compLev = '012'


if '0' in compLev:
    subproc.check_call([PATH+'./configure', '--with-coord=cylindrical',Integrator, '--enable-fofc', methodGasOrMHD,\
                        METHOD, ORDER, problemToConfig])

#    subproc.check_call([PATH+'./configure', '--with-coord=cylindrical', methodGasOrMHD, METHOD, ORDER, problemToConfig])
    

if '1' in compLev:
    subproc.check_call(['make', 'clean'])

if '2' in compLev:
    subproc.check_call(['make', 'all', 'MACHINE=macosx'])

if '3' in compLev:
    subproc.check_call(['./athena', '-i', inputFile],  cwd = './bin' )


    ./configure --with-coord=cylindrical --with-problem=torus9 --with-gas=mhd --with-flux=hlld --with-order=2p --enable-mpi --with-integrator=vl
