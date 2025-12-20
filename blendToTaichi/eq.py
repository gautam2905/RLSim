import taichi as ti
import trimesh
import numpy as np

# Initialize Taichi
ti.init(arch=ti.gpu)

# --- CONFIGURATION ---
OBJ_FILENAME = "./try1.obj"  # Replace with your file name
# ---------------------

print(f"Loading {OBJ_FILENAME}...")

# 1. Load the mesh using trimesh
# force='mesh' ensures we get a mesh object even if the file contains a scene
mesh = trimesh.load(OBJ_FILENAME, force='mesh') 

# 2. Extract Data
# We need vertices (points) and faces (indices)
# We flatten the indices to a 1D array because that's what Taichi expects
np_vertices = mesh.vertices.astype(np.float32)
np_indices = mesh.faces.flatten().astype(np.int32)

# --- NORMALIZATION (Crucial Step) ---
# 3D models come in wild scales. This centers and scales the model 
# so it fits perfectly inside the Taichi view (box of -1 to 1).
bbox_min = np_vertices.min(axis=0)
bbox_max = np_vertices.max(axis=0)
center = (bbox_min + bbox_max) / 2
scale = np.linalg.norm(bbox_max - bbox_min)
np_vertices = (np_vertices - center) / scale 

# 3. Create Taichi Fields
num_vertices = len(np_vertices)
num_indices = len(np_indices)

vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
indices = ti.field(dtype=ti.i32, shape=num_indices)
colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)

# 4. Transfer data to GPU
vertices.from_numpy(np_vertices)
indices.from_numpy(np_indices)

# --- HANDLING COLORS/MATERIALS ---
# Taichi GGUI doesn't auto-load .mtl files.
# We will inspect the mesh visual properties from trimesh.
if hasattr(mesh.visual, 'vertex_colors'):
    # If the .obj has vertex colors, use them
    # trimesh colors are usually 0-255 (RGBA), we need 0.0-1.0 (RGB)
    v_colors = mesh.visual.vertex_colors[:, :3] / 255.0
    colors.from_numpy(v_colors.astype(np.float32))
    print("Loaded vertex colors from mesh.")
else:
    # Fallback: Give it a nice default color (e.g., Orange)
    colors.fill(ti.Vector([1.0, 0.7, 0.2])) 
    print("No vertex colors found, using default orange.")

# 5. Set up the Visualization Window (GGUI)
window = ti.ui.Window("Taichi OBJ Viewer", (1024, 768), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# Position camera to look at center
camera.position(0, 0, 2)
camera.lookat(0, 0, 0)

while window.running:
    # Update camera controls (WASD + Mouse)
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    
    # Setup light so it's not dark
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))

    # Render the mesh
    # per_vertex_color uses the 'colors' field we set up earlier
    scene.mesh(vertices, indices=indices, per_vertex_color=colors)
    
    canvas.scene(scene)
    window.show()