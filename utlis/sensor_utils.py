import mujoco
import numpy as np
from create_env import Environment_presets
from mujoco.viewer import launch_passive
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go

def add_cube_collision_pairs(self):
    # Find the body named "cube" in the environment
    cube_body = None
    for body in self.arena.worldbody.find_all('body'):
        if 'cube' in str(body):
            cube_body = body
            break

    if cube_body is None:
        return

    geoms1 = cube_body.find_all('geom')

    for geom1 in geoms1:
        if (hasattr(geom1, 'conaffinity') and geom1.conaffinity != 0) or \
                (hasattr(geom1, 'dclass') and 'collision' in str(geom1.dclass)):

            # Iterate over all other bodies
            for body2 in self.arena.worldbody.find_all('body'):
                if body2 == cube_body:  # Avoid self-collisions
                    continue
                if "cube" in str(body2):
                    continue

                # Skip bodies attached to "iiwa"
                if 'iiwa' in str(body2):
                    continue

                geoms2 = body2.find_all('geom')

                for geom2 in geoms2:
                    if (hasattr(geom2, 'conaffinity') and geom2.conaffinity != 0) or \
                            (hasattr(geom2, 'dclass') and 'collision' in str(geom2.dclass)):

                        try:
                            self.arena.contact.add(element_name="pair", geom1=geom1, geom2=geom2)
                        except Exception as e:
                            print(f"Error adding pair: {e}")


def get_snesor_grid(env):
    #env = Environment_presets()
    collision_file = "compute_roboSDF/collision_points.txt"

    # Check if the collision points file exists
    if os.path.exists(collision_file):
        print(f"Collision points file found. Loading from {collision_file}.")

        collision_points = np.loadtxt(collision_file)
        fig = go.Figure(data=[go.Scatter3d(
            x=collision_points[::, 0],
            y=collision_points[::, 1],
            z=collision_points[::, 2],
            mode='markers',
            marker=dict(
                size=1,
                color='red',  # Set the color of the markers
                opacity=0.8
            )
        )])
#
        # Set the layout of the plot
        fig.update_layout(
            title="Collision Points - Loaded from File",
            scene=dict(
                xaxis_title='X axis',
                yaxis_title='Y axis',
                zaxis_title='Z axis'
            )
        )
#
        # Show the interactive plot
        fig.show(renderer='browser')

        return collision_points

    #env.generate_Tesla()
    xml_cube = "environment_parts/cube/cube.xml"
    cube = env.get_model_from_xml(xml_cube)

    #Shelf = env.get_model_from_xml('environment_parts/Regal_braun/regal_braun.xml')
    #env.add_model_to_arena(Shelf, 'shelf', [0, -0.8, 0], None, is_mocap=False)

    env.add_model_to_arena(cube, 'cube', [0, 0, 5], None, is_mocap=True)

    add_cube_collision_pairs(env)

    env.compile_model()
    #env.display_environment()
    # Load the MuJoCo model and data
    model = env.model
    data = env.data
    # env.data =set_cube_position(data, np.array([0,2,0]))
    # env.display_environment()
    # Define the grid resolution and size for the cube's movement
    grid_resolution = 0.01  # The step size for moving the cube
    grid_min = np.array([-2.0, -2.0, 0.1])  # Min bounds of the grid (adjust based on your environment)
    grid_max = np.array([2, 2, 2.0])  # Max bounds of the grid (adjust based on your environment)

    # Define cube properties (position, size, etc.)
    cube_size = np.array([grid_resolution, grid_resolution, grid_resolution])  # Size of the cube (adjust as needed)
    collision_points = []  # List to store collision points

    # Helper function to check for collisions
    def check_collision(model, data):
        data.qpos[:] = 0
        mujoco.mj_forward(model, data)  # Perform forward dynamics to update positions and collision detection
        for i in range(data.ncon):
            con = data.contact[i]
            if "iiwa" in model.geom(con.geom1).name or "iiwa" in model.geom(con.geom2).name:
                return False
            else:
                # print(f"Collision detected between {model.geom(con.geom1).name} and {model.geom(con.geom2).name}")
                return True
        return False  # No collision

    # Main loop: move the cube through the grid
    x_range = np.arange(grid_min[0], grid_max[0], grid_resolution)
    y_range = np.arange(grid_min[1], grid_max[1], grid_resolution)
    z_range = np.arange(grid_min[2], grid_max[2], grid_resolution)

    for x in tqdm(x_range):
        for y in y_range:
            for z in z_range:
                # Set the cube position
                cube_pos = np.array([x, y, z])
                data = set_cube_position(model,data, cube_pos)

                # Check for collisions
                in_collision = check_collision(model, data)
                if in_collision:
                    # with launch_passive(env.model, env.data) as viewer:
                    #   # set_camera(viewer)
                    #   while viewer.is_running():
                    #       mujoco.mj_forward(env.model, env.data)
                    #       viewer.sync()

                    collision_points.append(cube_pos)

    # Convert collision points to a numpy array for easier processing
    collision_points = np.array(collision_points)

    # Save or visualize the 3D collision map
    np.savetxt("compute_roboSDF/collision_points.txt", collision_points)  # Save the points to a file
    print(f"Collision points saved. Total points: {len(collision_points)}")

    # Optionally, visualize the collision points using matplotlib

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(collision_points[:, 0], collision_points[:, 1], collision_points[:, 2], c='r', marker='o')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()
    return collision_points
# Helper function
# to set the cube's position
def set_cube_position(model,data, position):
    cube_body_id = model.body('cube').mocapid[0] # Get the ID of the cube body
    data.mocap_pos[cube_body_id] = position
    return data

