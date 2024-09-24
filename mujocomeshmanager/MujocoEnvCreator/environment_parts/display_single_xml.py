import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_assembly import CustomEnvironment

env = CustomEnvironment()

# Load XML files
xml_file_cage = "environment_parts/UR_gripper_modified/ur_gripper_modified.xml"
model_cage = env.get_model_from_xml(xml_file_cage)
env.add_model_to_arena(model_cage, "Regal", [0, 0, 0], None, is_mocap=False)

env.display_environment()