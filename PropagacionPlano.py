import numpy as np
import matplotlib.pyplot as plt

def propagacion_fuego(
    size=100,
    init_value=0.5,
    steps=200,
    D=0.25,
    dx=1.0,
    dy=1.0,
    dt=None,               # si None, se calcula CFL estable para difusión 2D
    reaction_coef=0.3,     # 0: sin reacción; >0: R(F)=k*F*(1-F)
    plot_every=25,
    show_plot=True
):
    """
    Difusión (+ reacción opcional) 2D con esquema explícito y frontera Neumann.

    F^{n+1}_{ij} = F^n_{ij}
                   + dt * [ D*(d2F/dx2 + d2F/dy2) + R(F^n_{ij}) ]

    - Estado inicial: grilla 0 con centro = init_value (4 celdas si size par).
    - Frontera: Neumann (flujo cero) SIEMPRE.
    - Colormap con autoescalado (vmax dinámico) para visualizar bien la propagación.
    """

    # ---- 1) Estado inicial
    F = np.zeros((size, size), dtype=float)
    mid = size // 2
    if size % 2 == 0:
        i0, i1 = mid - 1, mid
        j0, j1 = mid - 1, mid
    else:
        i0 = i1 = mid
        j0 = j1 = mid
    F[i0:i1+1, j0:j1+1] = init_value

    # ---- 2) Paso de tiempo estable (CFL) para difusión 2D
    if dt is None:
        # dt <= 1 / (2*D*(1/dx^2 + 1/dy^2)) – usamos margen 0.9
        dt = 0.9 / (2.0 * D * (1.0/(dx*dx) + 1.0/(dy*dy)))

    # ---- 3) Frontera Neumann (flujo cero)
    def frontera_neumann(A):
        A[0,   :] = A[1,   :]
        A[-1,  :] = A[-2,  :]
        A[:,   0] = A[:,   1]
        A[:,  -1] = A[:,  -2]
        return A

    # ---- 4) Plot inicial con autoescalado
    if show_plot:
        plt.figure(figsize=(5, 5))
        vmax0 = max(1e-6, F.max())     # evita vmax=0
        im = plt.imshow(F, origin='lower', cmap='hot', vmin=0, vmax=vmax0)
        plt.colorbar(label='Intensidad (autoescala)')
        plt.title("t = 0")
        plt.grid(True)
        plt.pause(0.001)

    # ---- 5) Bucle temporal (FOR)
    for n in range(1, steps + 1):
        # aplicar frontera antes de derivar
        F = frontera_neumann(F)

        # Laplaciano interior
        Fc = F[1:-1, 1:-1]
        Fxx = (F[2:, 1:-1]   - 2.0*Fc + F[:-2, 1:-1]) / (dx*dx)
        Fyy = (F[1:-1, 2:]   - 2.0*Fc + F[1:-1, :-2]) / (dy*dy)
        lap = Fxx + Fyy

        # Reacción opcional: R(F)=k F (1-F)
        if reaction_coef > 0.0:
            R = reaction_coef * Fc * (1.0 - Fc)
        else:
            R = 0.0

        # Paso explícito
        F_new = F.copy()
        F_new[1:-1, 1:-1] = Fc + dt * (D * lap + R)

        # Frontera + límites físicos
        F_new = frontera_neumann(F_new)
        F_new = np.clip(F_new, 0.0, 1.0)
        F = F_new

        # refrescar gráfico con autoescalado
        if show_plot and (n % plot_every == 0 or n == steps):
            vmax = max(1e-6, F.max())
            im.set_data(F)
            im.set_clim(0, vmax)      # <- autoescala para que no se vea “todo negro”
            plt.title(f"t = {n*dt:.3f} (paso {n}/{steps})")
            plt.pause(0.5)

    if show_plot:
        plt.show()

    return F

propagacion_fuego()
