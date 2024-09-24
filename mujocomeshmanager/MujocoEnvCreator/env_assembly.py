from dm_control import mjcf
from dm_control import mujoco
from mujoco.viewer import launch_passive
from scipy.spatial.transform import Rotation as R

try:
    from .logger import logger
    from .utils.environment_utils import ChangeDirectoryMeta
except:
    import os 
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from logger import logger
    from utils.environment_utils import ChangeDirectoryMeta
    

class CustomEnvironment(metaclass=ChangeDirectoryMeta):
    def __init__(self, model_name = None):
        self.data = None
        self.model = None
        self.models = []
        self.sim = None
        self.viewer = None
        self.arena = mjcf.RootElement(model_name)
        self._call_history = []
        self.bodies_attached = []
        self.add_default_ground()

    def add_default_ground(self):
        xml_lab = "environment_parts/laboratory/laboratory.xml"
        lab_model = self.get_model_from_xml(xml_lab)
        self.add_model_to_arena(lab_model, "ground", [0, 0, 0], None, is_mocap=False)

    def add_all_col_pairs(self):
        logger.info(f"Add all collision pairs of number of bodies: {len(self.bodies_attached)}")
        for i in range(len(self.bodies_attached)):
            for j in range(i + 1, len(self.bodies_attached)):
                body1 = self.bodies_attached[i]
                body2 = self.bodies_attached[j]
                geoms1 = body1.find_all('geom')
                geoms2 = body2.find_all('geom')

                for geom1 in geoms1:
                    if (hasattr(geom1, 'conaffinity') and geom1.conaffinity != 0) or \
                            (hasattr(geom1, 'dclass') and 'collision' in str(geom1.dclass)):

                        for geom2 in geoms2:
                            if (hasattr(geom2, 'conaffinity') and geom2.conaffinity != 0) or \
                                    (hasattr(geom2, 'dclass') and 'collision' in str(geom2.dclass)):

                                try:
                                    self.arena.contact.add(element_name="pair", geom1=geom1, geom2=geom2)
                                except Exception as e:
                                    print(f"Error adding pair: {e}")

    def load_preset(self, path_to_mjb):
        """Loads an MJCB binary file from the given path."""
        logger.info(f"Load preset from {path_to_mjb} and generate mj model and data")
        self.model = mujoco.MjModel.from_binary_path(path_to_mjb)
        self.data = mujoco.MjData(self.model)

    def save_preset(self, file_name):
        """Saves an MJCB binary file into the presets' directory."""
        logger.info(f"Save model to presets/{file_name}")
        if self.model is None:
            logger.warning("MODEL NOT YET COMPILED")
            self.compile_model()
        mujoco.mj_saveModel(self.model, f"presets/{file_name}.mjb")

    def export_with_assets(self, model_name, path):
        logger.info(f"Save model to {path}/presets/{model_name}")
        mjcf.export_with_assets(self.arena, f"{path}/presets/{model_name}", out_file_name=f"{model_name}.xml")


    def get_model_from_xml(self, xml_path):
        """Loads an MJCF XML file into the environment."""
        logger.info(f"Get model from xml path: {xml_path}")
        with open(xml_path, 'r') as file:
            xml_content = file.read()
        model = mjcf.from_xml_string(xml_content)
        return model
    def add_model_to_site(self, mj_model, site_name, quat=None):
        site_attachment = self.arena.find('site', site_name)
        if quat is not None:
            # check if site has orientation specified
            rotation = R.from_quat(quat)
            if site_attachment.quat is not None:
                site_attachment.quat = quat
            if site_attachment.axisangle is not None:
                site_attachment.axisangle = rotation.as_rotvec()
            if site_attachment.euler is not None:
                site_attachment.euler = rotation.as_euler('zyx')
        site_attachment.attach(mj_model)

    def add_model_to_arena(self, mj_model, body_name, pos: list, quat: list = None, is_mocap=False):
        logger.info(f"add model to arena: {body_name}, pos: {pos}, quat: {quat}, Mocap: {is_mocap}")
        mocap_body = self.arena.worldbody.add('body', name=body_name, mocap=is_mocap)
        mocap_body.pos = pos
        if quat is not None:
            mocap_body.quat = quat
        site_name = f"{mocap_body.name}_site_attach"
        mocap_body.add('site', name=site_name)
        mocap_station_attach = self.arena.find('site', site_name)
        mocap_station_attach.attach(mj_model)
        # self.bodies_attached.append(mj_model)

    def compile_model(self):
        logger.info(f"Compile current arena into a model")
        physics = mujoco.Physics.from_xml_string(
            xml_string=self.arena.to_xml_string(),
            assets=self.arena.get_assets())

        self.model = physics.model.ptr
        self.data = physics.data.ptr
        return self.model, self.data

    def display_environment(self):
        """Displays the current simulation environment."""
        if self.model is None:
            logger.warning("MODEL NOT YET COMPILED")
            self.compile_model()
        with launch_passive(self.model, self.data) as viewer:
            #set_camera(viewer)
            while viewer.is_running():
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

    def generate_iiwas_with_ur_grippers(self):
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

        xml_gripper_robot_right = "environment_parts/UR_gripper/ur_gripper.xml"
        gripper_robot_right = self.get_model_from_xml(xml_gripper_robot_right)
        self.add_model_to_site(gripper_robot_right, 'cage/iiwa_1/gripper_attachment')

        xml_gripper_robot_left = "environment_parts/UR_gripper/ur_gripper.xml"
        gripper_robot_left = self.get_model_from_xml(xml_gripper_robot_left)
        self.add_model_to_site(gripper_robot_left, 'cage/iiwa/gripper_attachment')


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
    env.export_with_assets('showcase_xml_export', '.')
    # Display the environment
    env.display_environment()
