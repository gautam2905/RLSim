import struct

# Simple STL writer function
def write_stl(filename, triangles, name="solid"):
    with open(filename, 'w') as f:
        f.write(f"solid {name}\n")
        for tri in triangles:
            f.write("facet normal 0 0 0\n")
            f.write("outer loop\n")
            for vertex in tri:
                f.write(f"vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
            f.write("endloop\n")
            f.write("endfacet\n")
        f.write(f"endsolid {name}\n")

# Helper to make a box (cuboid)
def make_box(x_min, x_max, y_min, y_max, z_min, z_max):
    # 8 corners
    v = [
        [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min], # Bottom
        [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]  # Top
    ]
    # 12 triangles (2 per face)
    indices = [
        [0,1,5], [0,5,4], # Front
        [1,2,6], [1,6,5], # Right
        [2,3,7], [2,7,6], # Back
        [3,0,4], [3,4,7], # Left
        [4,5,6], [4,6,7], # Top
        [3,2,1], [3,1,0]  # Bottom
    ]
    return [[v[i] for i in idx] for idx in indices]

# --- PARAMETERS (Based on the paper) ---
# Mold Size: 200mm wide, 800mm tall, 35mm thick (17.5mm half-thickness [cite: 225])
mold_width = 0.2    # 200mm
mold_height = 0.8   # 800mm
mold_thick = 0.035  # 35mm

# Brass Plate: 0.5mm thick [cite: 5]
plate_thick = 0.0005 

# SEN (Nozzle): Simple square pipe for now
sen_width = 0.04
sen_depth = 0.3 # How deep it goes in

# --- GENERATING ---

# 1. MOLD WALLS (The main box)
# Centered at x=0, z=0. Top at y=0, growing downwards.
mold_tris = make_box(-mold_width/2, mold_width/2, -mold_height, 0, -mold_thick/2, mold_thick/2)
write_stl("constant/triSurface/mold_walls.stl", mold_tris, "mold")

# 2. BRASS PLATES (Two thin sheets on the wide walls)
# Front Plate
plate_front = make_box(-mold_width/2, mold_width/2, -mold_height, 0, mold_thick/2, mold_thick/2 + plate_thick)
# Back Plate
plate_back = make_box(-mold_width/2, mold_width/2, -mold_height, 0, -mold_thick/2 - plate_thick, -mold_thick/2)
write_stl("constant/triSurface/brass_plates.stl", plate_front + plate_back, "plates")

# 3. SEN (The Nozzle)
# A pipe sticking in the top
sen_tris = make_box(-sen_width/2, sen_width/2, -sen_depth, 0.1, -sen_width/2, sen_width/2)
write_stl("constant/triSurface/SEN.stl", sen_tris, "SEN")

print("SUCCESS: 3 STL files created in constant/triSurface/")