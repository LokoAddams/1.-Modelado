import geopandas as gdp
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import  FuncAnimation
from scipy.interpolate import griddata

## Este codigo es muy similar al original proporcionado por el docente
## la unica diferencia es que en la visualizacion el terreno es plano
## ya que la parte de la simulacion no cambia, por que aunque en el codigo proporcionado la vizualizacion
## es en 3d, no se toma en cuenta las alturas para la simulacion
gdf = gdp.read_file("elevaciones.gpkg")
if "altura" in gdf.columns:
    elev_col = "altura"
else:
    elev_col = gdf.columns[-1]
print(gdf["altura"])
print(gdf.geometry)

coords =  np.array([[geom.centroid.x, geom.centroid.y ] for geom in gdf.geometry], dtype=np.float64)
x = coords[:, 0]
y = coords[:, 1]


nx, ny = 60, 60
xi = np.linspace(x.min(), x.max(), nx)
yi = np.linspace(y.min(), y.max(), ny)
X, Y = np.meshgrid(xi, yi)
Z = np.zeros((nx, ny))

input()
################### simulacion
T = np.zeros_like(Z)
cx, cy = nx // 2, ny // 2
T[cx-2:cx+3, cy-2:cy+3] = 1000
D = 1.5
alpha = 0.1
Tign = 10
dt= 0.1

def laplacian(T):
    return (np.roll(T, 1, 0) + np.roll(T, -1, 0) + np.roll(T, 1, 1) + np.roll(T, -1, 1) - 4 * T)

def get_Tem(T):
    S = (T > Tign).astype(float) *30.0## 1 o 0 * (30.0)
    dTdt = D * laplacian(T) - alpha * T + S
    return T + dTdt * dt
################### visualizacion

fig = pl.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

np.set_printoptions(precision=2, suppress=True, linewidth=200, threshold=np.prod(T.shape))


def animate(frame):
    global T
    T = get_Tem(T)
    ax.clear()
    ax.set_zlim(Z.min(), Z.max() + 10)
    ax.plot_surface(X,Y,Z,cmap="terrain", alpha=0.9, linewidth=0)
    Thorn = np.clip(T / 1000, 0, 1)
    print("---------------------------------------------------------------")
    print(np.round(T, 2))
    print("---------------------------------------------------------------")
    ax.plot_surface(X,Y,Z+0.5, facecolors=pl.cm.hot(Thorn), alpha=0.7)
ani = FuncAnimation(fig, animate, frames=120, interval=120, blit=False)
pl.show()