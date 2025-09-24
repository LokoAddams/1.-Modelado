import geopandas as gdp
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata

# =========================
# Configuración de archivos
# =========================
ELEV_GPKG = "elevaciones.gpkg"   # capa de elevación
HUM_GPKG  = "humedades.gpkg"     # capa con IDs/clases (humedales, etc.)
HUM_FIELD = None                 # si conoces el campo exacto, ej. "codigo"; si None, se detecta

# ================
# Utilidades
# ================
def pick_value_column(gdf, prefer=None):
    """Elige una columna numérica para usar como valor."""
    if prefer and prefer in gdf.columns:
        return prefer
    numeric_cols = [c for c in gdf.columns
                    if c != "geometry" and np.issubdtype(gdf[c].dtype, np.number)]
    if numeric_cols:
        return numeric_cols[-1]
    # última no-geometry como fallback
    return [c for c in gdf.columns if c != "geometry"][-1]


def interpolate_to_grid(gdf, value_col, xi, yi, method="cubic", fill_value=0.0):
    """Interpola columna a grilla (xi, yi) con griddata."""
    coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf.geometry], dtype=np.float64)
    x = coords[:, 0]; y = coords[:, 1]
    v = gdf[value_col].astype(float).to_numpy()
    X, Y = np.meshgrid(xi, yi)
    V = griddata((x, y), v, (X, Y), method=method)
    # relleno de huecos
    mask = np.isnan(V)
    if mask.any():
        V_near = griddata((x, y), v, (X, Y), method="nearest")
        V[mask] = V_near[mask]
    V = np.nan_to_num(V, nan=fill_value)
    return X, Y, V

def normalize_ids_to_01(A):
    """Normaliza matriz de IDs/códigos a [0,1] por rango (min->0, max->1)."""
    A = A.astype(float)
    amin, amax = float(np.nanmin(A)), float(np.nanmax(A))
    if amax > amin:
        A2 = (A - amin) / (amax - amin)
    else:
        A2 = np.zeros_like(A)
    return np.clip(A2, 0.0, 1.0)

# ===================
# Lectura del terreno
# ===================
gdf_elev = gdp.read_file(ELEV_GPKG)
elev_col = pick_value_column(gdf_elev, prefer="altura")
coords_e = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf_elev.geometry], dtype=np.float64)
x = coords_e[:, 0]; y = coords_e[:, 1]
z = gdf_elev[elev_col].astype(float).to_numpy()

nx, ny = 60, 60
xi = np.linspace(x.min(), x.max(), nx)
yi = np.linspace(y.min(), y.max(), ny)
X, Y = np.meshgrid(xi, yi)
Z = griddata((x, y), z, (X, Y), method="cubic")
Z = np.nan_to_num(Z, nan=0.0)

# ======================
# Lectura de HUMEDAD (IDs/clases)
# ======================
gdf_hum = gdp.read_file(HUM_GPKG)
hum_col = pick_value_column(gdf_hum, prefer=HUM_FIELD) # escoger campo numérico

# IMPORTANTE: para IDs/clases usar 'nearest' (no cubic) para evitar overshoot
_, _, H_ids = interpolate_to_grid(gdf_hum, hum_col, xi, yi, method="nearest", fill_value=0.0)

# Normalizar IDs a [0,1] (0 = menos húmedo / menor ID, 1 = más húmedo / mayor ID)
H = normalize_ids_to_01(H_ids)

# Sanity check
# print("H stats -> min:", H.min(), "max:", H.max(), "any<0?", (H<0).any(), "any>1?", (H>1).any())

# ---------------- Simulación ----------------
T = np.zeros_like(Z)
cx, cy = nx // 2, ny // 2
T[cx-2:cx+3, cy-2:cy+3] = 1000  # Foco inicial

D = 0.4
alpha = 0.01
Tign = 200
dt = 0.1

# --- Mapa de combustibles (ejemplo) ---
C_map = np.zeros_like(Z, dtype=int)
C_map[10:20, :] = 1
C_map[35:50, :] = 2
C_map[20:35, :] = 1
C_map[40:45, 15:45] = 1

fuel_params = {0: 0.0, 1: 50.0, 2: 25.0}
R_map = np.zeros_like(Z)
for fuel_type, heat_value in fuel_params.items():
    R_map[C_map == fuel_type] = heat_value

def laplacian(Z_):
    return (np.roll(Z_, 1, 0) + np.roll(Z_, -1, 0) + np.roll(Z_, 1, 1) + np.roll(Z_, -1, 1) - 4 * Z_)

def get_Tem(T_):
    # La humedad H (fija) frena ignición local: más humedad -> menor S
    S = (T_ > Tign).astype(float) * (R_map * (1 - H))
    dT = D * laplacian(T_) - alpha * T_ + S
    return T_ + dT * dt

# ---------------- Visualización ----------------
fig = pl.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

np.set_printoptions(precision=2, suppress=True, linewidth=200, threshold=np.prod(T.shape))

def animate(frame):
    global T
    T = get_Tem(T)
    ax.clear()

    # Terreno
    ax.plot_surface(X, Y, Z, cmap="terrain", alpha=0.9, linewidth=0)

    # Fuego
    Thorn = np.clip(T / 1000, 0, 1)
    ax.plot_surface(X, Y, Z + 10, facecolors=pl.cm.hot(Thorn), alpha=0.7)

    # Humedad (ya normalizada 0..1)
    ax.plot_surface(X, Y, Z - 5, facecolors=pl.cm.Blues(H), alpha=0.4)

    # (opcional) ver T/H por consola
    # print("Frame", frame, "T min/max:", T.min(), T.max())
    print("HUMEDAD: ")
    print(H)

ani = FuncAnimation(fig, animate, frames=120, interval=120, blit=False)
pl.show()
