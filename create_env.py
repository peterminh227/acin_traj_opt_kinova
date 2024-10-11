import ast

import numpy as np
import mujoco
import time
from mujocomeshmanager.MujocoEnvCreator.env_assembly import CustomEnvironment


class Environment_presets(CustomEnvironment):
    def __init__(self, model_name=None):
        super().__init__(model_name)

    def generate_Pascal_Pernoulli(self, gripper=False):
        # Load XML files
        xml_file_cage = "environment_parts/cage/cage.xml"
        model_cage = self.get_model_from_xml(xml_file_cage)
        self.add_model_to_arena(model_cage, "cage", [0, 0, 0], None, is_mocap=False)

        xml_robot_left = "environment_parts/iiwa_arc/iiwa.xml"
        iiwa_left = self.get_model_from_xml(xml_robot_left)
        self.add_model_to_site(iiwa_left, 'cage/robot_attach_left')

        xml_robot_right = "environment_parts/iiwa_arc/iiwa.xml"
        iiwa_right = self.get_model_from_xml(xml_robot_right)
        self.add_model_to_site(iiwa_right, 'cage/robot_attach_right')

        if gripper:
            xml_gripper_robot_right = "environment_parts/UR_gripper/ur_gripper.xml"
            gripper_robot_right = self.get_model_from_xml(xml_gripper_robot_right)
            self.add_model_to_site(gripper_robot_right, 'cage/iiwa_1/gripper_attachment')

            xml_gripper_robot_left = "environment_parts/UR_gripper/ur_gripper.xml"
            gripper_robot_left = self.get_model_from_xml(xml_gripper_robot_left)
            self.add_model_to_site(gripper_robot_left, 'cage/iiwa/gripper_attachment')

    def generate_Tesla(self, gripper=False):
        # Load XML files
        xml_socket = "environment_parts/iiwa_socket/iiwa_socket.xml"
        socket = self.get_model_from_xml(xml_socket)
        self.add_model_to_arena(socket, 'socket', [0, 0, 0], None, is_mocap=False)

        xml_robot = "environment_parts/iiwa_arc/iiwa.xml"
        iiwa = self.get_model_from_xml(xml_robot)
        self.add_model_to_site(iiwa, 'iiwa_socket/robot_attach')

        if gripper:
            xml_gripper = "environment_parts/UR_gripper/ur_gripper.xml"
            gripper = self.get_model_from_xml(xml_gripper)
            self.add_model_to_site(gripper, 'iiwa_socket/iiwa/gripper_attachment')


    def generate_iros_env(self):
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
    env = Environment_presets()
    env.generate_Tesla()
    # Compile model
    env.compile_model()
    #env.save_preset('robothub')
    #env.export_with_assets('robohub_xml_export', os.path.abspath('.'))
    env.display_environment()
