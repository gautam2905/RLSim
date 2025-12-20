import taichi as ti

# Initialize Taichi
ti.init(arch=ti.gpu)

# --- Simulation Parameters ---
RES_X = 200        # Resolution (Width of billet)
RES_Y = 600        # Resolution (Length of strand)
dt = 1e-2          # Time step
dx = 1.0           # Grid spacing

# --- Metal Properties (Steel-like) ---
alpha = 0.8        # Thermal diffusivity
casting_speed = 20.0 # Speed of the strand moving down
T_liquid = 1500.0  # Pouring temperature
T_solidus = 1300.0 # Solidification temperature
T_ambient = 300.0  # Cooling water temperature
cooling_rate = 0.08 # Intensity of surface cooling

# --- Fields ---
# T stores the temperature at every grid point
T = ti.field(dtype=ti.f32, shape=(RES_X, RES_Y))
# pixels stores the RGB color for visualization
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))

@ti.kernel
def init():
    # Initialize the entire strand to ambient first (just to start)
    for i, j in T:
        T[i, j] = T_ambient

@ti.kernel
def step():
    # Loop over all grid points (excluding strict boundaries for simple finite difference)
    for i, j in T:
        if j == RES_Y - 1:
            # INLET: Top boundary is always fresh hot liquid steel
            T[i, j] = T_liquid
        elif j > 0 and i > 0 and i < RES_X - 1:
            # PHYSICAL EQUATION: dT/dt + v * dT/dy = alpha * Laplacian(T)
            
            # 1. Diffusion (Heat conduction inside the metal)
            # Laplacian in 2D
            laplacian = (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1] - 4*T[i, j]) / (dx * dx)
            
            # 2. Advection (The metal moving downwards)
            # We use upwind scheme for stability: (T[current] - T[upstream])
            dT_dy = (T[i, j+1] - T[i, j]) / dx
            
            # Combine them: New_T = Old_T + dt * (Diffusion + Advection)
            T[i, j] += dt * (alpha * laplacian + casting_speed * dT_dy)

    # --- Boundary Conditions (Surface Cooling) ---
    for j in range(RES_Y):
        # Left Wall Cooling
        flux_L = cooling_rate * (T[0, j] - T_ambient)
        T[0, j] -= dt * flux_L
        
        # Right Wall Cooling
        flux_R = cooling_rate * (T[RES_X-1, j] - T_ambient)
        T[RES_X-1, j] -= dt * flux_R

@ti.kernel
def render_colors():
    for i, j in pixels:
        temp = T[i, j]
        
        # Color mapping:
        # > T_solidus = Red (Liquid)
        # < T_solidus = Blue/Gray (Solid)
        
        if temp > T_solidus:
            # Gradient Red for liquid
            intensity = (temp - T_solidus) / (T_liquid - T_solidus)
            pixels[i, j] = ti.Vector([1.0, 1.0 - intensity * 0.5, 0.0]) # Yellow-ish Red
        else:
            # Gradient Blue for solid
            intensity = (T_solidus - temp) / 1000.0
            pixels[i, j] = ti.Vector([0.0, 0.0, 1.0 - intensity])

# --- Main GUI Loop ---
gui = ti.GUI("Continuous Casting Simulation (Taichi)", res=(RES_X, RES_Y))
init()

while gui.running:
    # Run multiple physics steps per frame for speed
    for _ in range(20):
        step()
    
    render_colors()
    gui.set_image(pixels)
    gui.show()