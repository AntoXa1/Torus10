#!/Users/dora/Library/Enthought/Canopy_32bit/User/bin/python

from scipy import *
from numpy import ndarray, zeros
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from matplotlib import pyplot as plt
from os import path



file1 = path.dirname(path.abspath(__file__))


file1 = file1 + '/'+ 'athena_plot.tmp'
print('file=', file1)
f1 = open(file1)
print("python from athena")

# READING CORE
s1 = f1.readline().split()

res =s1[0]

N1= int(s1[0])
N2= int(s1[1])

print("N1=", N1,    "N2=", N2)

A = zeros((N1, N2))

line=f1.read().splitlines()


js= 0
je =N1
for i in range(N2):

    A[:, i] = map(float, line[js : je])

    js = js + N1+1
    je = je + N1+1

f1.close()

if (N2 == 1) :
#    x1 = linspace(1, N1+1)
    x1 = range(1, N1+1)
#    print (x1, A[:,0])
    fig =plot(x1, A[:, 0], '*')

if(N2 <> 1 ) :

#    A = log10( A )

    ind1 = linspace(1, N1,  N1)
    ind2 = linspace(1, N2,  N2)
    X, Y = meshgrid(ind2, ind1)



#    pcolormesh(X,Y, A,  edgecolors="None"); show(); exit()


    fig = plt.figure()
    ax = Axes3D(fig)

#    print(A)

    ax.plot_wireframe(X, Y, A, rstride=2, cstride=4 )

plt.show()

exit()
