import geopandas as gdp
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata

# lectura del mapa
gdf = gdp.read_file("elevaciones.gpkg")
if "altura" in gdf.columns:
    elev_col = "altura"
else:
    elev_col = gdf.columns[-1]

coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf.geometry], dtype=np.float64)
x = coords[:, 0]
y = coords[:, 1]
z = gdf[elev_col].astype(float).to_numpy()

nx, ny = 60, 60
xi = np.linspace(x.min(), x.max(), nx)
yi = np.linspace(y.min(), y.max(), ny)
X, Y = np.meshgrid(xi, yi)
Z = griddata((x, y), z, (X, Y), method="cubic")
Z = np.nan_to_num(Z, nan=0.0) # Reemplaza los NaN con 0.0
# ---------------- Simulación ----------------
T = np.zeros_like(Z)
cx, cy = nx // 2, ny // 2
T[cx-2:cx+3, cy-2:cy+3] = 1000  # Foco inicial

D = 2.5
alpha = 0.01
Tign = 10
dt = 0.1
# --- Nivel 3: Inicio de la Implementación ---

# 1. Creamos un mapa de tipos de combustible (ej. 0=roca, 1=pasto, 2=bosque)
#    Aquí creo unas bandas aleatorias para simularlo.
C_map = np.zeros_like(Z, dtype=int)
C_map[10:20, :] = 1  # Una banda de "pasto"
C_map[35:50, :] = 2  # Una banda de "bosque"
C_map[20:35, :] = 1  # Una banda de "bosque"
C_map[40:45, 15:45] = 1 # Un parche de pasto dentro del bosque

# 2. Definimos los parámetros de la fuente de calor para cada combustible
fuel_params = {
    0: 0.0,   # Roca: no genera calor
    1: 50.0,  # Pasto: genera calor intenso
    2: 25.0   # Bosque: genera calor moderado
}

# 3. Creamos la matriz de Tasa de Reacción (R_map) a partir del mapa de combustible
R_map = np.zeros_like(Z)
for fuel_type, heat_value in fuel_params.items():
    R_map[C_map == fuel_type] = heat_value

# --- Nivel 3: Fin de la Implementación ---

# --- humedad: matriz aleatoria entre 0 y 1 ---
H = np.random.rand(nx, ny)  # 0 = seco, 1 = muy húmedo

def laplacian(Z):
    return (np.roll(Z, 1, 0) + np.roll(Z, -1, 0) + np.roll(Z, 1, 1) + np.roll(Z, -1, 1) - 4 * Z)

def get_Tem(T):
    # Ignición ajustada por humedad
    S = (T > Tign).astype(float) * (R_map * (1 - H))  # más humedad → menos ignición
    dT = D * laplacian(T) - alpha * T + S
    return T + dT * dt

# ---------------- Visualización ----------------
fig = pl.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

np.set_printoptions(precision=2, suppress=True, linewidth=200, threshold=np.prod(T.shape))


def animate(frame):
    global T
    T = get_Tem(T)
    ax.clear()

    # Terreno
    ax.plot_surface(X, Y, Z, cmap="terrain", alpha=0.9, linewidth=0)

    # Fuego
    Thorn = np.clip(T / 1000, 0, 1)
    ax.plot_surface(X, Y, Z+10, facecolors=pl.cm.hot(Thorn), alpha=0.7)
    print("---------------------------------------------------------------")
    print(np.round(T, 2))
    print("---------------------------------------------------------------")
    # Humedad (azuladas)
    ax.plot_surface(X, Y, Z-5, facecolors=pl.cm.Blues(H), alpha=0.4)

ani = FuncAnimation(fig, animate, frames=120, interval=120, blit=False)
pl.show()
