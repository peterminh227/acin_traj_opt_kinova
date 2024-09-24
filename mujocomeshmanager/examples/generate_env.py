import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mujoco_creator.env_assembly import CustomEnvironment

# Example usage
if __name__ == "__main__":
    env = CustomEnvironment()
    #env.load_preset("presets/cage_kukas_URgrippers.mjb")
    #env.display_environment()

    # Load XML files
    xml_file_cage = "environment_parts/cage/cage.xml"
    model_cage = env.get_model_from_xml(xml_file_cage)
    env.add_model_to_arena(model_cage, "cage", [0, 0, 0], None, is_mocap=False)

    xml_robot_left = "environment_parts/iiwa_arc/iiwa.xml"
    iiwa_left = env.get_model_from_xml(xml_robot_left)
    env.add_model_to_site(iiwa_left, 'cage/robot_attach_left')

    xml_robot_right = "environment_parts/iiwa_arc/iiwa.xml"
    iiwa_right = env.get_model_from_xml(xml_robot_right)
    env.add_model_to_site(iiwa_right, 'cage/robot_attach_right')


    xml_gripper_robot_right = "environment_parts/UR_gripper/ur_gripper.xml"
    gripper_robot_right = env.get_model_from_xml(xml_gripper_robot_right)
    env.add_model_to_site(gripper_robot_right, 'cage/iiwa_1/gripper_attachment')

    xml_gripper_robot_left = "environment_parts/UR_gripper/ur_gripper.xml"
    gripper_robot_left = env.get_model_from_xml(xml_gripper_robot_left)
    env.add_model_to_site(gripper_robot_left, 'cage/iiwa/gripper_attachment')

    # Compile model
    env.compile_model()
    #env.save_preset('cage_kukas_noGrippers')
    env.export_with_assets('showcase_xml_export', os.path.abspath('.'))
    # Display the environment
    env.display_environment()