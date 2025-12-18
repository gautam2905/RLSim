import taichi as ti
import numpy as np

# Initialize Taichi
# using gpu for speed, fall back to cpu if gpu fails
try:
    ti.init(arch=ti.gpu)
except:
    ti.init(arch=ti.cpu)

# --- 1. CONFIGURATION ---
@ti.data_oriented
class Config:
    def __init__(self):
        # Grid & Time
        self.res_x = 256   # Slab Thickness resolution
        self.res_y = 600   # Casting Length resolution
        self.dx = 0.002    # 2mm grid cell size
        self.dt = 0.1    # Time step (Conservative for stability)
        
        # Steel Properties (Low Carbon Steel)
        self.rho = 7200.0       # Density kg/m3
        self.T_liquidus = 1530.0
        self.T_solidus = 1480.0
        self.L_fusion = 270000.0 # Latent Heat (J/kg)
        self.Cp = 650.0          # Specific Heat
        self.k_liquid = 35.0     # Conductivity
        self.k_solid = 30.0
        
        # Simulation Parameters
        self.da_C = 2.5e6        # Drag coefficient (Higher = stiffer solid)
        self.mold_length = 0.8   # Length of the copper mold (m)

cfg = Config()

# --- 2. THERMAL SOLVER (Enthalpy Method) ---
@ti.data_oriented
class ThermalSolver:
    def __init__(self):
        # Fields: Temperature, Enthalpy, Liquid Fraction, Temp buffer
        self.T = ti.field(dtype=ti.f32, shape=(cfg.res_x, cfg.res_y))
        self.H = ti.field(dtype=ti.f32, shape=(cfg.res_x, cfg.res_y)) 
        self.f_l = ti.field(dtype=ti.f32, shape=(cfg.res_x, cfg.res_y))
        self.temp_H = ti.field(dtype=ti.f32, shape=(cfg.res_x, cfg.res_y)) 

    @ti.func
    def get_T_from_H(self, enthalpy: float) -> float:
        # Converts Enthalpy back to Temperature handling Latent Heat
        H_solid = cfg.Cp * cfg.T_solidus
        H_liquid = H_solid + cfg.L_fusion
        temp = 0.0
        if enthalpy < H_solid:
            temp = enthalpy / cfg.Cp
        elif enthalpy > H_liquid:
            temp = cfg.T_liquidus + (enthalpy - H_liquid) / cfg.Cp
        else:
            # Mushy Zone: Interpolate
            f_l = (enthalpy - H_solid) / cfg.L_fusion
            temp = cfg.T_solidus + f_l * (cfg.T_liquidus - cfg.T_solidus)
        return temp

    @ti.func
    def update_liquid_fraction(self, i: int, j: int):
        temp = self.T[i, j]
        if temp >= cfg.T_liquidus:
            self.f_l[i, j] = 1.0
        elif temp <= cfg.T_solidus:
            self.f_l[i, j] = 0.0
        else:
            self.f_l[i, j] = (temp - cfg.T_solidus) / (cfg.T_liquidus - cfg.T_solidus)

    @ti.func
    def sample_H(self, i: float, j: float) -> float:
        # SAFE SAMPLING: Clamps coordinates to valid range to prevent crashes
        i = ti.math.clamp(i, 0.0, float(cfg.res_x - 1.01))
        j = ti.math.clamp(j, 0.0, float(cfg.res_y - 1.01))
        
        i0, j0 = int(i), int(j)
        i1, j1 = i0 + 1, j0 + 1
        w_i, w_j = i - i0, j - j0
        
        return (1.0 - w_i) * (1.0 - w_j) * self.H[i0, j0] + \
               w_i * (1.0 - w_j) * self.H[i1, j0] + \
               (1.0 - w_i) * w_j * self.H[i0, j1] + \
               w_i * w_j * self.H[i1, j1]

    @ti.kernel
    def advect(self, v_field: ti.template()):
        # Semi-Lagrangian advection
        for i, j in self.H:
            pos = ti.Vector([float(i), float(j)])
            vel = v_field[i, j]
            # Backtrace
            prev_pos = pos - vel * cfg.dt / cfg.dx
            self.temp_H[i, j] = self.sample_H(prev_pos.x, prev_pos.y)
            
        # Copy back
        for i, j in self.H:
            self.H[i, j] = self.temp_H[i, j]

    @ti.kernel
    def diffuse_and_update(self):
        for i, j in self.H:
            # SAFETY CHECK: Only compute diffusion for interior points
            # This prevents accessing i-1 or i+1 when at the edge
            if i > 0 and i < cfg.res_x - 1 and j > 0 and j < cfg.res_y - 1:
                
                k = cfg.k_liquid if self.f_l[i, j] > 0.5 else cfg.k_solid
                alpha = k / cfg.rho # Simplified diffusivity
                
                # Standard Laplacian
                laplacian_T = (self.T[i+1, j] + self.T[i-1, j] + 
                               self.T[i, j+1] + self.T[i, j-1] - 
                               4*self.T[i, j]) / (cfg.dx**2)
                
                # Update Enthalpy
                self.H[i, j] += cfg.dt * (k * laplacian_T / cfg.rho)

            # Update Temp/Liquid Fraction based on new Enthalpy
            self.T[i, j] = self.get_T_from_H(self.H[i, j])
            self.update_liquid_fraction(i, j)

# --- 3. FLUID SOLVER (Navier-Stokes) ---
@ti.data_oriented
class FluidSolver:
    def __init__(self):
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=(cfg.res_x, cfg.res_y))
        self.new_v = ti.Vector.field(2, dtype=ti.f32, shape=(cfg.res_x, cfg.res_y))
        self.p = ti.field(dtype=ti.f32, shape=(cfg.res_x, cfg.res_y))
        self.new_p = ti.field(dtype=ti.f32, shape=(cfg.res_x, cfg.res_y))
        self.div = ti.field(dtype=ti.f32, shape=(cfg.res_x, cfg.res_y))

    @ti.func
    def sample_v(self, i: float, j: float) -> ti.Vector:
        # Clamped sampling for velocity
        i = ti.math.clamp(i, 0.0, float(cfg.res_x - 1.01))
        j = ti.math.clamp(j, 0.0, float(cfg.res_y - 1.01))
        i0, j0 = int(i), int(j)
        i1, j1 = i0 + 1, j0 + 1
        w_i, w_j = i - i0, j - j0
        
        return (1.0 - w_i) * (1.0 - w_j) * self.v[i0, j0] + \
               w_i * (1.0 - w_j) * self.v[i1, j0] + \
               (1.0 - w_i) * w_j * self.v[i0, j1] + \
               w_i * w_j * self.v[i1, j1]

    @ti.kernel
    def advect_velocity(self):
        for i, j in self.v:
            pos = ti.Vector([float(i), float(j)])
            vel = self.v[i, j]
            prev_pos = pos - vel * cfg.dt / cfg.dx 
            self.new_v[i, j] = self.sample_v(prev_pos.x, prev_pos.y)
        
        for i, j in self.v:
            self.v[i, j] = self.new_v[i, j]

    @ti.kernel
    def apply_forces(self, f_l_field: ti.template()):
        for i, j in self.v:
            gravity = ti.Vector([0.0, -9.81])
            fl = f_l_field[i, j]
            
            # Carman-Kozeny Drag (Stops flow in solid regions)
            epsilon = 1e-4
            drag = 0.0
            if fl < 0.99:
                 drag = -cfg.da_C * ((1.0 - fl)**2) / (fl**3 + epsilon)
            
            # Apply forces
            self.v[i, j] += (gravity + drag * self.v[i, j]) * cfg.dt

    @ti.kernel
    def compute_divergence(self):
        for i, j in self.div:
            # Only compute for interior
            if i > 0 and i < cfg.res_x - 1 and j > 0 and j < cfg.res_y - 1:
                self.div[i, j] = 0.5 * (self.v[i+1, j].x - self.v[i-1, j].x + 
                                        self.v[i, j+1].y - self.v[i, j-1].y) / cfg.dx
            else:
                self.div[i, j] = 0.0

    @ti.kernel
    def pressure_jacobi(self):
        for i, j in self.p:
            if i > 0 and i < cfg.res_x - 1 and j > 0 and j < cfg.res_y - 1:
                self.new_p[i, j] = (self.p[i+1, j] + self.p[i-1, j] + 
                                    self.p[i, j+1] + self.p[i, j-1] - 
                                    self.div[i, j] * cfg.dx**2) * 0.25
            else:
                self.new_p[i, j] = self.p[i, j]

    @ti.kernel
    def update_p(self):
        for i, j in self.p:
            self.p[i, j] = self.new_p[i, j]

    @ti.kernel
    def apply_projection(self):
        for i, j in self.v:
             if i > 0 and i < cfg.res_x - 1 and j > 0 and j < cfg.res_y - 1:
                grad_p = ti.Vector([self.p[i+1, j] - self.p[i-1, j],
                                    self.p[i, j+1] - self.p[i, j-1]]) * 0.5 / cfg.dx
                self.v[i, j] -= grad_p

# --- 4. MAIN SIMULATOR ---
@ti.data_oriented
class CasterSimulator:
    def __init__(self):
        self.thermal = ThermalSolver()
        self.fluid = FluidSolver()
        self.initialize_field()
        
    @ti.kernel
    def initialize_field(self):
        # Start with a dummy bar (cold)
        for i, j in self.thermal.T:
            self.thermal.T[i, j] = 300.0
            self.thermal.H[i, j] = cfg.Cp * 300.0
            self.thermal.f_l[i, j] = 0.0
            self.fluid.v[i, j] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def apply_boundary_conditions(self):
        # 1. INLET (Submerged Entry Nozzle)
        # We enforce this AFTER diffusion to ensure the inlet stays hot
        center = cfg.res_x // 2
        half_w = 15 # Nozzle width
        
        # We inject at y = res_y - 5 to be safely inside the domain
        y_inj = cfg.res_y - 5
        
        for x in range(center - half_w, center + half_w):
            for y_off in range(0, 4): # Force a small block of inlet
                y = y_inj + y_off
                self.fluid.v[x, y] = ti.Vector([0.0, -0.8]) # Jet velocity
                self.thermal.T[x, y] = 1560.0
                self.thermal.H[x, y] = cfg.Cp * 1560.0 + cfg.L_fusion
                self.thermal.f_l[x, y] = 1.0

        # 2. WALL COOLING (Left and Right)
        for j in range(cfg.res_y):
            # Calculate physical height (0 is bottom)
            h_m = j * cfg.dx
            
            # Zonal Cooling Coefficients
            h_coeff = 100.0 # Air (Bottom)
            
            # Top of the domain (Mold)
            top_m = cfg.res_y * cfg.dx
            if h_m > top_m - cfg.mold_length:
                h_coeff = 2500.0 # Mold (High cooling)
            elif h_m > top_m * 0.4:
                h_coeff = 800.0  # Water Spray (Medium)

            # Left Wall (i=0)
            # Only cool if it's NOT the inlet region
            if j < cfg.res_y - 10:
                T_L = self.thermal.T[0, j]
                flux_L = h_coeff * (T_L - 300.0)
                # Apply cooling to Enthalpy
                self.thermal.H[0, j] -= (flux_L * cfg.dt / (cfg.rho * cfg.dx))

                # Right Wall (i=res_x-1)
                T_R = self.thermal.T[cfg.res_x-1, j]
                flux_R = h_coeff * (T_R - 300.0)
                self.thermal.H[cfg.res_x-1, j] -= (flux_R * cfg.dt / (cfg.rho * cfg.dx))

    def step(self):
        # 1. Fluid
        self.fluid.advect_velocity()
        self.fluid.apply_forces(self.thermal.f_l)
        
        # Pressure Projection Loop (Python controlled)
        self.fluid.compute_divergence()
        for _ in range(20): 
            self.fluid.pressure_jacobi()
            self.fluid.update_p()
        self.fluid.apply_projection()
        
        # 2. Thermal
        self.thermal.advect(self.fluid.v)
        self.thermal.diffuse_and_update()
        
        # 3. BCs
        self.apply_boundary_conditions()

# --- 5. RUN ---
sim = CasterSimulator()
gui = ti.GUI("Safe Continuous Casting 2D", res=(cfg.res_x, cfg.res_y))

print("Simulation Started. Red = Liquid, Blue = Solid. Wait for the stream to fall...")

while gui.running:
    # Run 10 physics steps per frame
    for _ in range(10):
        sim.step()
    
    # Visualization
    # Pixels: Red (Liquid) -> Blue (Solid)
    # We create a simple color map based on Liquid Fraction (f_l)
    fl_np = sim.thermal.f_l.to_numpy()
    
    # Create an RGB image
    # R = f_l (Liquid is Red)
    # B = 1 - f_l (Solid is Blue)
    img = np.zeros((cfg.res_x, cfg.res_y, 3), dtype=np.float32)
    img[:, :, 0] = fl_np        # Red channel
    img[:, :, 2] = 1.0 - fl_np  # Blue channel
    
    gui.set_image(img)
    gui.show()