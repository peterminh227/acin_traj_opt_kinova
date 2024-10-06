from create_env import Environment_presets
from mujoco.viewer import launch_passive
import mujoco
import numpy as np
import torch
from compute_roboSDF.SDF_utils import KUKAiiwaLayer, BPSDF
from utlis.sensor_utils import get_snesor_grid
from torch.cuda.amp import GradScaler, autocast
from mujoco import mjx
from dm_control import mjcf
import jax
import jax.numpy as jnp


env = Environment_presets()
env.generate_Tesla()

Shelf = env.get_model_from_xml('environment_parts/Regal_braun/regal_braun.xml')
env.add_model_to_arena(Shelf, 'shelf', [0, -0.8, 0], None, is_mocap=False)
# Compile model
env.compile_model()
#DEBUG display model
#env.display_environment()

#test configuration in collision
bounds_right = {
    'lower': np.array([-2.9670597283903604,
                       -2.0943951023931953,
                       -2.9670597283903604,
                       -2.0943951023931953,
                       -2.9670597283903604,
                       -2.0943951023931953,
                       -3.0543261909900767]),
    'upper': np.array([2.9670597283903604,
                       2.0943951023931953,
                       2.9670597283903604,
                       2.0943951023931953,
                       2.9670597283903604,
                       2.0943951023931953,
                       3.0543261909900767])
}

# Number of configurations to generate
num_configs = 1

# Generate random configurations within the bounds
qpos_collision_backboard = np.random.uniform(bounds_right['lower'], bounds_right['upper'], (num_configs, len(bounds_right['lower']))).tolist()

#qpos_collision_backboard = [[-1.5,1.5,0,0,0,0,0],
#                            [-1.,1.5,0,0,0,0,0],
#                            [-1.,1.,0,0,0,0,0],
#                            [-1., 1., 0, 0.05, 0, -0., 0],#nocol
#                            [-1.,1.,0,0.1,0,-0.,0],         #nocol
#                            [-1.,1.,0,0.1,0,-0.25,0],
#                            [-1.,1.,0,2,0,0,0],
#                            [-1.,1.,0,2,0,2,0],
#                            [-1.,0,0,2,0,2,0],
#                            [0.,0,0,2,0,2,0],
#                            [0.,0,0,2,0,2,0],
#                            ]

#with launch_passive(env.model, env.data) as viewer:
#    # set_camera(viewer)
#    while viewer.is_running():
#        for q in qpos_collision_backboard:
#            env.data.qpos[:] = np.array(q)
#            mujoco.mj_forward(env.model, env.data)
#            viewer.sync()

#TEST torch collision

device = 'cuda'
q_pos = torch.tensor(qpos_collision_backboard).float().to(device).reshape(-1,7)
H0_base = np.identity(4)
H0_base[0:3,3] = np.array([0,0,0.25])
robot_offset_to_world = torch.from_numpy(H0_base).unsqueeze(0).to(device).expand(len(q_pos),4,4).float()
#
kuka = KUKAiiwaLayer(device).to(device)
bp_sdf  = BPSDF(n_func=24,domain_min=-1,domain_max=1,robot=kuka,device=device)
model = torch.load(f'compute_roboSDF/BP_iiwa_24.pt')
sensor_map = get_snesor_grid(env)
#sensor_map = sensor_map[:5]
batch_size = len(sensor_map)
#batch_size = 2000
#
#
total_sum = np.zeros(len(qpos_collision_backboard))  # Initialize the total sum
sensor_map_torch = torch.from_numpy(sensor_map).to(device).float()

import time
start_time = time.time()
sdf,ana_grad,idx = bp_sdf.get_whole_body_sdf_batch_mod(
    sensor_map_torch,
    robot_offset_to_world,
    q_pos,
    model,
    use_derivative=False,
    used_links=[0, 1, 2, 3, 4, 5, 6, 7]
)
duration = time.time() -start_time
print(f"SDF computation processed in {duration:.4f} seconds")

sdf_cpu = sdf.cpu().numpy()
minimum_values = sdf_cpu.min(axis=1)
modified_values = np.where(minimum_values > 0, 0.0, minimum_values)


'''
for i in range(0, len(sensor_map), batch_size):
   batch = sensor_map[i:i + batch_size]
   print(f"Processing batch {i // batch_size + 1}, Shape: {batch.shape}")
   sensor_map_torch = torch.from_numpy(batch).to(device).float()

   with autocast():  # Enable mixed precision
       sdf,ana_grad,idx = bp_sdf.get_whole_body_sdf_batch_new(
           sensor_map_torch,
           robot_offset_to_world,
           q_pos,
           model,
           use_derivative=False,
           used_links=[0, 1, 2, 3, 4, 5, 6, 7]
       )
       sdf_cpu = sdf.cpu().numpy()
       minimum_values = sdf_cpu.min(axis=1)
       modified_values = np.where(minimum_values > 0, 0.0, minimum_values)
       total_sum += modified_values


   del sensor_map_torch, sdf, ana_grad
   torch.cuda.empty_cache()
'''   
print("Debug value = ",modified_values)


exit()

with open('model_mjx/test.xml', 'r') as file:
    xml_content = file.read()
model = mujoco.MjModel.from_xml_string(xml_content)
#mjcf.export_with_assets(env.arena, f"model_mjx/", out_file_name=f"test.xml")
mjx_model = mjx.put_model(model)


@jax.vmap
def batched_collision_check(pos):
    mjx_data = mjx.make_data(mjx_model)
    qpos = mjx_data.qpos.at[:].set(pos)  # Set all elements of qpos
    mjx_data = mjx_data.replace(qpos=qpos)

    # Perform collision checking with the updated position
    dist = mjx.forward(mjx_model, mjx_data).contact.dist

    collision_array = jax.numpy.array(dist)
    return collision_array


pos_jax = jnp.array(qpos_collision_backboard)
import time
start_time = time.time()
collision_results = jax.jit(batched_collision_check)(pos_jax)
end_time = time.time()
negative_sums = np.zeros(collision_results.shape[0])


# Calculate the sum of negative values for each row
for i in range(collision_results.shape[0]):
    negative_sums[i] = np.sum(collision_results[i][collision_results[i] < 0])  # Sum only negative values in the row
print(f"Time taken: {end_time - start_time:.6f} seconds")
print("")
