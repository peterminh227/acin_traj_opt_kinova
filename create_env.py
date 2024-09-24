import ast

import numpy as np
import mujoco
import time
from mujocomeshmanager.MujocoEnvCreator.env_assembly import CustomEnvironment


def create_iros_env():
    env = CustomEnvironment()

    xml_file_table = "environment_parts/TU_table_nice/TU_table_nice.xml"
    model_table = env.get_model_from_xml(xml_file_table)
    env.add_model_to_arena(model_table, "table", pos=[0, 0, 0], quat=None, is_mocap=False)

    xml_file_kinova = "environment_parts/kinova3/kinova3.xml"
    model_kinova = env.get_model_from_xml(xml_file_kinova)
    env.add_model_to_site(model_kinova, 'tu_tisch_verstellbar/table_mount_kinova')

    xml_file_gripper= "environment_parts/UR_gripper/ur_gripper.xml"
    gripper = env.get_model_from_xml(xml_file_gripper)
    env.add_model_to_site(gripper, 'tu_tisch_verstellbar/kinova3/gripper_mount')

    return env


if __name__ == '__main__':
    env = create_iros_env()
    
    # Compile model
    env.compile_model()
    env.save_preset('robothub')
    env.export_with_assets('robohub_xml_export', os.path.abspath('.'))
    env.display_environment()
