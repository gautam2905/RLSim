import taichi as ti
import trimesh
import numpy as np
import os

# Initialize Taichi
ti.init(arch=ti.gpu)

# --- CONFIGURATION ---
OBJ_FILENAME = "./try1.obj"  # <--- Make sure this matches your file
# ---------------------

if not os.path.exists(OBJ_FILENAME):
    raise FileNotFoundError(f"Could not find {OBJ_FILENAME}. Is it in the same folder?")

print(f"Loading {OBJ_FILENAME}...")

# 1. Load Mesh
# force='mesh' merges all objects in the file into one single mesh
mesh = trimesh.load(OBJ_FILENAME, force='mesh')

# --- ERROR CHECK / FIX 1: Material Conversion ---
# If the mesh uses a texture image (.mtl with map_Kd), trimesh loads it as 'TextureVisuals'.
# This helper forces it to convert that into simple per-vertex colors so we can see it.
try:
    mesh.visual = mesh.visual.to_color()
except Exception as e:
    print(f"Warning: Could not convert visuals to color. ({e})")

# 2. Extract Data (and ensure 32-bit floats for Taichi)
np_vertices = mesh.vertices.astype(np.float32)
np_indices = mesh.faces.flatten().astype(np.int32)

# --- ERROR CHECK / FIX 2: Normals ---
# We need normals for lighting to look correct. 
# If the file doesn't have them, trimesh can calculate them.
if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
    print("Computing normals...")
    mesh.compute_vertex_normals()
np_normals = mesh.vertex_normals.astype(np.float32)

# --- NORMALIZATION ---
# Center and scale the model to fit in the view
bbox_min = np_vertices.min(axis=0)
bbox_max = np_vertices.max(axis=0)
center = (bbox_min + bbox_max) / 2
scale = np.linalg.norm(bbox_max - bbox_min)
np_vertices = (np_vertices - center) / scale

# 3. Create Taichi Fields
num_vertices = len(np_vertices)
num_indices = len(np_indices)

verts_field = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
norms_field = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices) # New: Normals
indices_field = ti.field(dtype=ti.i32, shape=num_indices)
colors_field = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)

# 4. Transfer to GPU
verts_field.from_numpy(np_vertices)
norms_field.from_numpy(np_normals)
indices_field.from_numpy(np_indices)

# Handle Colors
if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) > 0:
    # Trimesh gives RGBA (0-255), Taichi wants RGB (0.0-1.0)
    v_colors = mesh.visual.vertex_colors[:, :3] / 255.0
    colors_field.from_numpy(v_colors.astype(np.float32))
    print("Colors loaded successfully.")
else:
    colors_field.fill(ti.Vector([0.8, 0.8, 0.8])) # Default Grey
    print("Using default color.")

# 5. Visualization Window
window = ti.ui.Window("Robust Taichi Viewer", (1024, 768), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# Initial camera position
camera.position(0, 0.5, 2.0)
camera.lookat(0, 0, 0)

print("Viewer running. Use WASD + Mouse Right Click to move.")

while window.running:
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    
    # Lighting (requires normals to work properly)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.3, 0.3, 0.3))

    # Render
    # We now pass 'normals' explicitly
    scene.mesh(verts_field, 
               indices=indices_field, 
               normals=norms_field, 
               per_vertex_color=colors_field)
    
    canvas.scene(scene)
    window.show()