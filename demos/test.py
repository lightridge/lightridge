import numpy;
from matplotlib import pyplot;
from matplotlib import cm;
from mpl_toolkits.mplot3d import Axes3D;
pyplot.interactive(True);

# Creat mesh.
X = numpy.arange(-1, 1, 0.1);
Y = numpy.arange(-1, 1, 0.1);
X, Y = numpy.meshgrid(X, Y);

# Create some data to plot.
A = numpy.copy(X);
B = numpy.copy(Y);
C = numpy.sqrt(X**2 + Y**2);
D = numpy.cos(C);
# Normalize data for colormap use.
A -= numpy.min(A); A /= numpy.max(A);
B -= numpy.min(B); B /= numpy.max(B);
C -= numpy.min(C); C /= numpy.max(C);
D -= numpy.min(D); D /= numpy.max(D);

# Create flat surface.
Z = numpy.zeros_like(X);

# Plot
fig = pyplot.figure(figsize=(12,6));
ax = fig.gca(projection='3d');
ax.plot_surface(X, Z, Y, rstride=1, cstride=1, facecolors = cm.coolwarm(A));
ax.plot_surface(X, Z+0.3, Y, rstride=1, cstride=1, facecolors = cm.coolwarm(B));
ax.plot_surface(X, Z+0.6, Y, rstride=1, cstride=1, facecolors = cm.coolwarm(C));
ax.plot_surface(X, Z+3, Y, rstride=1, cstride=1, facecolors = cm.coolwarm(D));
fig.savefig("test.pdf")
