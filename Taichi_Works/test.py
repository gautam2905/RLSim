import taichi as ti
import numpy as np

# Initialize Taichi (mapped to CUDA if available, else CPU)
ti.init(arch=ti.gpu)

# --- Configuration & Constants ---
RES_X, RES_Y = 400, 800  # Resolution mimicking the mold aspect ratio
STEPS = 40               # Substeps per frame
viscosity = 0.02         # Kinematic viscosity (nu)
omega = 1.0 / (3.0 * viscosity + 0.5) # Relaxation parameter

# LBM D2Q9 Constants
w = ti.field(dtype=float, shape=9)
e = ti.Vector.field(2, dtype=int, shape=9)

# --- Fields ---
# Distribution functions (f_old -> f_new for double buffering)
f_old = ti.field(dtype=float, shape=(RES_X, RES_Y, 9))
f_new = ti.field(dtype=float, shape=(RES_X, RES_Y, 9))

# Macroscopic variables
rho = ti.field(dtype=float, shape=(RES_X, RES_Y))
vel = ti.Vector.field(2, dtype=float, shape=(RES_X, RES_Y))
image = ti.Vector.field(3, dtype=float, shape=(RES_X, RES_Y))

# Geometry Mask: 0 = Fluid, 1 = Solid (Walls/SEN), 2 = Inlet, 3 = Outlet
mask = ti.field(dtype=int, shape=(RES_X, RES_Y))

# --- Setup LBM Basis ---
@ti.kernel
def init_constants():
    # Weights for D2Q9
    w[0] = 4.0 / 9.0
    for i in range(1, 5): w[i] = 1.0 / 9.0
    for i in range(5, 9): w[i] = 1.0 / 36.0
    
    # Basis vectors
    e[0] = [0, 0]
    e[1], e[2], e[3], e[4] = [1, 0], [0, 1], [-1, 0], [0, -1]
    e[5], e[6], e[7], e[8] = [1, 1], [-1, 1], [-1, -1], [1, -1]

# --- Geometry Definition ---
# Recreating the SEN and Mold from
@ti.kernel
def init_geometry():
    SEN_WIDTH = 40
    SEN_TIP_DEPTH = 300
    PORT_HEIGHT = 40
    
    for i, j in mask:
        # Default: Fluid
        mask[i, j] = 0
        
        # 1. Outer Walls (Mold Faces)
        if i < 5 or i > RES_X - 5 or j > RES_Y - 5:
            mask[i, j] = 1 # Solid wall
        
        # 2. Bottom Outlet (Open boundary)
        if j < 5:
            mask[i, j] = 3 # Outlet

        # 3. Submerged Entry Nozzle (SEN) Geometry
        # The central pipe coming from top
        if abs(i - RES_X // 2) < SEN_WIDTH // 2 and j > (RES_Y - SEN_TIP_DEPTH):
            mask[i, j] = 1 # Solid SEN walls
            
            # The inner bore (Inlet)
            if abs(i - RES_X // 2) < (SEN_WIDTH // 2 - 6):
                mask[i, j] = 2 # Inlet fluid
                
            # The Side Ports - Critical for the "Jet" pattern
            # We cut holes in the side of the SEN near the bottom
            is_near_bottom = j < (RES_Y - SEN_TIP_DEPTH + PORT_HEIGHT + 10) and j > (RES_Y - SEN_TIP_DEPTH + 10)
            if is_near_bottom and abs(i - RES_X // 2) < SEN_WIDTH // 2:
                mask[i, j] = 0 # Fluid (The Port)
        
        # 4. The SEN "Cup" bottom (stops flow going straight down)
        if abs(i - RES_X // 2) < SEN_WIDTH // 2 and j < (RES_Y - SEN_TIP_DEPTH + 10) and j > (RES_Y - SEN_TIP_DEPTH):
             mask[i, j] = 1 # Solid Cup Bottom

    # Initialize f with rest density
    for i, j, k in f_old:
        f_old[i, j, k] = w[k]
        f_new[i, j, k] = w[k]

# --- LBM Core ---
@ti.kernel
def collide_and_stream():
    for i, j in ti.ndrange(RES_X, RES_Y):
        # 1. Read macroscopic
        current_f = ti.Vector([0.0]*9)
        for k in range(9):
            current_f[k] = f_old[i, j, k]
        
        local_rho = 0.0
        local_vel = ti.Vector([0.0, 0.0])
        
        for k in range(9):
            local_rho += current_f[k]
            local_vel += current_f[k] * e[k]
        
        if local_rho > 0:
            local_vel /= local_rho
        
        # Force inlet velocity at SEN top
        if mask[i, j] == 2:
            local_vel = ti.Vector([0.0, -0.35]) # Downward injection
            local_rho = 1.0

        # Equilibrium
        u_sq = local_vel.norm_sqr()
        for k in range(9):
            eu = local_vel.dot(e[k])
            f_eq = w[k] * local_rho * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*u_sq)
            
            # BGK Collision
            current_f[k] = (1.0 - omega) * current_f[k] + omega * f_eq
            
        # Streaming (Push to neighbors in f_new)
        if mask[i, j] != 1: # If not solid
            for k in range(9):
                ni, nj = i + e[k][0], j + e[k][1]
                if 0 <= ni < RES_X and 0 <= nj < RES_Y:
                    # Bounce-back handling for walls
                    if mask[ni, nj] == 1:
                        # Reverse direction index mapping for D2Q9
                        # 0->0, 1->3, 2->4, 3->1, 4->2 ...
                        inv_k = 0
                        if k == 1: inv_k = 3
                        elif k == 2: inv_k = 4
                        elif k == 3: inv_k = 1
                        elif k == 4: inv_k = 2
                        elif k == 5: inv_k = 7
                        elif k == 6: inv_k = 8
                        elif k == 7: inv_k = 5
                        elif k == 8: inv_k = 6
                        
                        # Streaming to solid bounces back to current cell
                        f_new[i, j, inv_k] = current_f[k]
                    else:
                        # Stream to neighbor
                        f_new[ni, nj, k] = current_f[k]

@ti.kernel
def update_macro():
    for i, j in ti.ndrange(RES_X, RES_Y):
        local_rho = 0.0
        local_vel = ti.Vector([0.0, 0.0])
        for k in range(9):
            local_rho += f_new[i, j, k]
            local_vel += f_new[i, j, k] * e[k]
            # Copy new to old for next step
            f_old[i, j, k] = f_new[i, j, k]
            
        rho[i, j] = local_rho
        if local_rho > 0:
            vel[i, j] = local_vel / local_rho

@ti.kernel
def render():
    for i, j in image:
        if mask[i, j] == 1:
            image[i, j] = [0.2, 0.2, 0.2] # Grey Walls
        elif mask[i, j] == 2:
            image[i, j] = [0.0, 0.5, 0.0] # Inlet Marker
        else:
            # Visualize Velocity Magnitude (mimicking Fig 6 in paper)
            v = vel[i, j].norm()
            # Heatmap: Blue (low) -> Red (High)
            image[i, j] = ti.Vector([v * 4.0, v * 2.0, 1.0 - v * 4.0])

# --- Main Loop ---
init_constants()
init_geometry()
gui = ti.GUI("CC Mold Flow (Structure Recreation)", res=(RES_X, RES_Y))

while gui.running:
    for _ in range(STEPS):
        collide_and_stream()
        update_macro()
    
    render()
    gui.set_image(image)
    gui.show()