import geopandas as gdp
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata

# ---------------- Lectura del mapa ----------------
gdf = gdp.read_file("elevaciones.gpkg")
elev_col = "altura" if "altura" in gdf.columns else gdf.columns[-1]

coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf.geometry], dtype=np.float64)
x = coords[:, 0]
y = coords[:, 1]
z = gdf[elev_col].astype(float).to_numpy()

nx, ny = 60, 60
xi = np.linspace(x.min(), x.max(), nx)
yi = np.linspace(y.min(), y.max(), ny)
X, Y = np.meshgrid(xi, yi)

Z = griddata((x, y), z, (X, Y), method="cubic")
Z = np.nan_to_num(Z, nan=0.0)  # reemplaza NaN por 0 para evitar huecos

# ---------------- Estado inicial ----------------
T = np.zeros_like(Z)
cx, cy = nx // 2, ny // 2
T[cx-2:cx+3, cy-2:cy+3] = 1000  # foco inicial

# ---------------- Parámetros ----------------
D0   = 0.4      # difusividad base (tu valor)
alpha = 0.01
Tign  = 200

# --- combustibles (siguen como los tenías) ---
C_map = np.zeros_like(Z, dtype=int)
C_map[10:20, :] = 1
C_map[20:35, :] = 1
C_map[35:50, :] = 2
C_map[40:45, 15:45] = 1

fuel_params = {0: 0.0, 1: 50.0, 2: 25.0}
R_map = np.zeros_like(Z, dtype=float)
for fuel_type, heat_value in fuel_params.items():
    R_map[C_map == fuel_type] = heat_value

# --- humedad (matriz 0..1) ---
H = np.random.rand(nx, ny)

# ---------------- Geometría y utilidades numéricas ----------------
dx = (X[0, -1] - X[0, 0]) / max(nx - 1, 1)
dy = (Y[-1, 0] - Y[0, 0]) / max(ny - 1, 1)

def grad_central_neumann(A):
    """Gradiente central con frontera tipo Neumann (reflejo)."""
    Ap = np.pad(A, ((1,1),(1,1)), mode='edge')
    dAdx = (Ap[1:-1, 2:] - Ap[1:-1, :-2]) / (2 * dx)
    dAdy = (Ap[2:, 1:-1] - Ap[:-2, 1:-1]) / (2 * dy)
    return dAdx, dAdy

# ---------------- Difusión dependiente de topografía (var. pequeña) ----------------
# Pendiente desde Z
dZdx, dZdy = grad_central_neumann(Z)
slope = np.hypot(dZdx, dZdy)

# Acotar extremos y normalizar para no desbalancear K
p_low, p_high = np.percentile(slope, [5, 95])
slope_clip = np.clip(slope, p_low, p_high)
slope_norm = (slope_clip - p_low) / (p_high - p_low + 1e-12)

# K(x,y) ≈ D0 * [0.95 .. 1.05]  (±5% alrededor de D0)
Kmin, Kmax = 0.95, 1.05
K_map = D0 * (Kmin + slope_norm * (Kmax - Kmin))
K_max_abs = float(K_map.max())

# dt estable (CFL difusión) con margen
dt_max = 0.2 * (min(dx, dy) ** 2) / (K_max_abs + 1e-12)
dt = min(0.1, dt_max)

def divergence_of_K_grad_neumann(T, K):
    """∇·(K ∇T) con promedios en caras y fronteras Neumann (reflejo)."""
    Tp = np.pad(T, ((1,1),(1,1)), mode='edge')
    Kp = np.pad(K, ((1,1),(1,1)), mode='edge')

    # K en caras
    K_e = 0.5 * (Kp[1:-1, 1:-1] + Kp[1:-1, 2:])   # este
    K_w = 0.5 * (Kp[1:-1, 1:-1] + Kp[1:-1, 0:-2]) # oeste
    K_n = 0.5 * (Kp[1:-1, 1:-1] + Kp[2:,   1:-1]) # norte
    K_s = 0.5 * (Kp[1:-1, 1:-1] + Kp[0:-2, 1:-1]) # sur

    # Gradientes hacia caras
    dT_e = (Tp[1:-1, 2:]   - Tp[1:-1, 1:-1]) / dx
    dT_w = (Tp[1:-1, 1:-1] - Tp[1:-1, 0:-2]) / dx
    dT_n = (Tp[2:,   1:-1] - Tp[1:-1, 1:-1]) / dy
    dT_s = (Tp[1:-1, 1:-1] - Tp[0:-2, 1:-1]) / dy

    # Divergencia
    div_x = (K_e * dT_e - K_w * dT_w) / dx
    div_y = (K_n * dT_n - K_s * dT_s) / dy
    return div_x + div_y

def get_Tem(T):
    # Fuente por ignición frenada por humedad (tu lógica)
    S = (T > Tign).astype(float) * (R_map * (1 - H))
    # Difusión topográfica suave
    diff = divergence_of_K_grad_neumann(T, K_map)
    dT = diff - alpha * T + S
    Tn = T + dT * dt
    return np.clip(Tn, 0.0, 1e6)

# ---------------- Visualización ----------------
fig = pl.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

def animate(frame):
    global T
    T = get_Tem(T)
    ax.clear()

    # Terreno
    ax.plot_surface(X, Y, Z, cmap="terrain", alpha=0.9, linewidth=0)

    # Fuego (coloreo con T normalizada)
    Thorn = np.clip(T / 1000.0, 0.0, 1.0)
    ax.plot_surface(X, Y, Z + 10, facecolors=pl.cm.hot(Thorn), alpha=0.7)

    # Humedad
    ax.plot_surface(X, Y, Z - 5, facecolors=pl.cm.Blues(H), alpha=0.4)

ani = FuncAnimation(fig, animate, frames=120, interval=120, blit=False)
pl.show()
