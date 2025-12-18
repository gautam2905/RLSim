import taichi as ti

# Initialize Taichi
ti.init(arch=ti.gpu)

# --- 1. CONFIGURATION ---
@ti.data_oriented
class Config:
    def __init__(self):
        # Grid & Time
        self.res_x = 256  # Width
        self.res_y = 600  # Height (Shortened slightly for performance)
        self.dx = 0.002   # m
        self.dt = 0.001   # s
        
        # Steel Properties
        self.rho = 7200.0       
        self.T_liquidus = 1530.0
        self.T_solidus = 1480.0
        self.L_fusion = 270000.0 
        self.Cp = 650.0          
        self.k_liquid = 35.0     
        self.k_solid = 30.0
        
        # Process
        self.da_C = 1.6e6        # Drag coefficient
        self.mold_length = 0.8   # m

cfg = Config()

# --- 2. THERMAL SOLVER ---
@ti.data_oriented
class ThermalSolver:
    def __init__(self):
        self.T = ti.field(dtype=ti.f32, shape=(cfg.res_x, cfg.res_y))
        self.H = ti.field(dtype=ti.f32, shape=(cfg.res_x, cfg.res_y)) 
        self.f_l = ti.field(dtype=ti.f32, shape=(cfg.res_x, cfg.res_y))
        self.temp_H = ti.field(dtype=ti.f32, shape=(cfg.res_x, cfg.res_y)) # Buffer for advection

    @ti.func
    def get_T_from_H(self, enthalpy: float) -> float:
        H_solid = cfg.Cp * cfg.T_solidus
        H_liquid = H_solid + cfg.L_fusion
        temp = 0.0
        if enthalpy < H_solid:
            temp = enthalpy / cfg.Cp
        elif enthalpy > H_liquid:
            temp = cfg.T_liquidus + (enthalpy - H_liquid) / cfg.Cp
        else:
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
        # Safe sampling with clamping
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
        for i, j in self.H:
            pos = ti.Vector([float(i), float(j)])
            vel = v_field[i, j]
            prev_pos = pos - vel * cfg.dt / cfg.dx
            self.temp_H[i, j] = self.sample_H(prev_pos.x, prev_pos.y)
            
        for i, j in self.H:
            self.H[i, j] = self.temp_H[i, j]

    @ti.kernel
    def diffuse_and_update(self):
        for i, j in self.H:
            # FIX: STRICT BOUNDARY CHECK
            if i > 0 and i < cfg.res_x - 1 and j > 0 and j < cfg.res_y - 1:
                k = cfg.k_liquid if self.f_l[i, j] > 0.5 else cfg.k_solid
                
                laplacian_T = (self.T[i+1, j] + self.T[i-1, j] + 
                               self.T[i, j+1] + self.T[i, j-1] - 
                               4*self.T[i, j]) / (cfg.dx**2)
                
                self.H[i, j] += cfg.dt * (k * laplacian_T / cfg.rho)

            # Update Temp/Liquid Fraction
            self.T[i, j] = self.get_T_from_H(self.H[i, j])
            self.update_liquid_fraction(i, j)

# --- 3. FLUID SOLVER ---
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
            gravity = ti.Vector([0.0, -9.8])
            fl = f_l_field[i, j]
            epsilon = 1e-4
            drag_coeff = -cfg.da_C * ((1.0 - fl)**2) / (fl**3 + epsilon)
            self.v[i, j] += (gravity + drag_coeff * self.v[i, j]) * cfg.dt

    @ti.kernel
    def compute_divergence(self):
        for i, j in self.div:
            if i > 0 and i < cfg.res_x - 1 and j > 0 and j < cfg.res_y - 1:
                self.div[i, j] = 0.5 * (self.v[i+1, j].x - self.v[i-1, j].x + 
                                        self.v[i, j+1].y - self.v[i, j-1].y) / cfg.dx

    @ti.kernel
    def pressure_jacobi(self):
        for i, j in self.p:
            if i > 0 and i < cfg.res_x - 1 and j > 0 and j < cfg.res_y - 1:
                self.new_p[i, j] = (self.p[i+1, j] + self.p[i-1, j] + 
                                    self.p[i, j+1] + self.p[i, j-1] - 
                                    self.div[i, j] * cfg.dx**2) * 0.25
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
        for i, j in self.thermal.T:
            self.thermal.T[i, j] = 300.0 # Ambient
            self.thermal.H[i, j] = cfg.Cp * 300.0
            self.thermal.f_l[i, j] = 0.0

    @ti.kernel
    def apply_boundary_conditions(self):
        # INLET
        center = cfg.res_x // 2
        half_w = 20
        for x in range(center - half_w, center + half_w):
            self.fluid.v[x, cfg.res_y-2] = ti.Vector([0.0, -0.5]) # Inject Down
            self.thermal.T[x, cfg.res_y-2] = 1560.0
            self.thermal.H[x, cfg.res_y-2] = cfg.Cp * 1560.0 + cfg.L_fusion
            self.thermal.f_l[x, cfg.res_y-2] = 1.0

        # WALL COOLING
        for j in range(cfg.res_y):
            # Height in meters from bottom
            h_m = j * cfg.dx
            
            # Simple Zonal Cooling Logic
            h_coeff = 200.0 # Default Air
            if h_m > (cfg.res_y * cfg.dx) - cfg.mold_length:
                h_coeff = 2000.0 # Mold
            elif h_m > 0.5:
                h_coeff = 800.0 # Spray

            # Left Wall
            T_L = self.thermal.T[0, j]
            flux_L = h_coeff * (T_L - 300.0)
            self.thermal.H[0, j] -= (flux_L * cfg.dt / (cfg.rho * cfg.dx))

            # Right Wall
            T_R = self.thermal.T[cfg.res_x-1, j]
            flux_R = h_coeff * (T_R - 300.0)
            self.thermal.H[cfg.res_x-1, j] -= (flux_R * cfg.dt / (cfg.rho * cfg.dx))

    def step(self):
        self.fluid.advect_velocity()
        self.fluid.apply_forces(self.thermal.f_l)
        
        self.fluid.compute_divergence()
        for _ in range(20):
            self.fluid.pressure_jacobi()
            self.fluid.update_p()
        self.fluid.apply_projection()
        
        self.thermal.advect(self.fluid.v)
        self.thermal.diffuse_and_update()
        self.apply_boundary_conditions()

# --- 5. RUN ---
sim = CasterSimulator()
gui = ti.GUI("Continuous Casting 2D", res=(cfg.res_x, cfg.res_y))

while gui.running:
    for _ in range(10):
        sim.step()
    
    # Visualization: 
    # Red = Liquid, Blue = Solid
    # We use f_l to blend colors
    pixels = sim.thermal.f_l.to_numpy()
    gui.set_image(pixels)
    gui.show()