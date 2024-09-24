import unittest
from unittest.mock import patch, mock_open, MagicMock
from MujocoEnvCreator.env_assembly import CustomEnvironment
from dm_control import mjcf
from dm_control import mujoco

class TestCustomEnvironment(unittest.TestCase):

    def setUp(self):
        self.env = CustomEnvironment()

    @patch('mujoco_env_creater.env_assembly.logger.info')
    @patch('mujoco_env_creater.env_assembly.mujoco.MjModel.from_binary_path')
    def test_load_preset(self, mock_from_binary_path, mock_logger_info):
        path_to_mjb = "test_preset.mjb"
        mock_model = MagicMock()
        mock_from_binary_path.return_value = mock_model
        mock_data = MagicMock()
        with patch('mujoco_env_creater.env_assembly.mujoco.MjData', return_value=mock_data):
            self.env.load_preset(path_to_mjb)
            mock_from_binary_path.assert_called_once_with(path_to_mjb)
            mock_logger_info.assert_called_with(f"Load preset from {path_to_mjb} and generate mj model and data")
            self.assertEqual(self.env.model, mock_model)
            self.assertEqual(self.env.data, mock_data)

    @patch('mujoco_env_creater.env_assembly.logger.info')
    @patch('mujoco_env_creater.env_assembly.mujoco.mj_saveModel')
    def test_save_preset(self, mock_mj_saveModel, mock_logger_info):
        mock_model = MagicMock()
        self.env.model = mock_model
        file_name = "test_preset"
        self.env.save_preset(file_name)
        mock_mj_saveModel.assert_called_once_with(mock_model, f"presets/{file_name}.mjb")
        mock_logger_info.assert_called_with(f"Save model to presets/{file_name}")

    @patch('mujoco_env_creater.env_assembly.logger.info')
    @patch('mujoco_env_creater.env_assembly.mjcf.export_with_assets')
    def test_export_with_assets(self, mock_export_with_assets, mock_logger_info):
        model_name = "test_model"
        self.env.export_with_assets(model_name, '.')
        mock_export_with_assets.assert_called_once_with(self.env.arena, f"./presets/{model_name}", out_file_name=f"{model_name}.xml")
        mock_logger_info.assert_called_with(f"Save model to ./presets/{model_name}")

    @patch('mujoco_env_creater.env_assembly.logger.info')
    @patch('builtins.open', new_callable=mock_open, read_data='<mujoco></mujoco>')
    def test_get_model_from_xml(self, mock_open, mock_logger_info):
        xml_path = "test_model.xml"
        model = self.env.get_model_from_xml(xml_path)
        mock_open.assert_called_once_with(xml_path, 'r')
        self.assertIsInstance(model, mjcf.RootElement)
        mock_logger_info.assert_called_with(f"Get model from xml path: {xml_path}")

    @patch('mujoco_env_creater.env_assembly.logger.info')
    def test_add_model_to_arena(self, mock_logger_info):
        mj_model = mjcf.RootElement()
        body_name = "test_body"
        pos = [0, 0, 0]
        quat = [1, 0, 0, 0]
        self.env.add_model_to_arena(mj_model, body_name, pos, quat)
        mock_logger_info.assert_called_with(f"add model to arena: {body_name}, pos: {pos}, quat: {quat}, Mocap: False")
        site_name = f"{body_name}_site_attach"
        site_attachment = self.env.arena.find('site', site_name)
        self.assertIsNotNone(site_attachment)

    @patch('mujoco_env_creater.env_assembly.logger.info')
    def test_compile_model(self, mock_logger_info):
        self.env.arena = mjcf.RootElement()
        mock_physics = MagicMock()
        with patch('mujoco_env_creater.env_assembly.mujoco.Physics.from_xml_string', return_value=mock_physics):
            model, data = self.env.compile_model()
            self.assertEqual(model, mock_physics.model.ptr)
            self.assertEqual(data, mock_physics.data.ptr)
        mock_logger_info.assert_called_with(f"Compile current arena into a model")

    @patch('mujoco_env_creater.env_assembly.launch_passive')
    @patch('mujoco_env_creater.env_assembly.logger.warning')
    @patch('mujoco_env_creater.env_assembly.mujoco.mj_step')
    def test_display_environment(self, mock_mj_step, mock_logger_warning, mock_launch_passive):
        mock_model = MagicMock()
        mock_data = MagicMock()
        self.env.model = mock_model
        self.env.data = mock_data
        mock_viewer = MagicMock()
        mock_launch_passive.return_value.__enter__.return_value = mock_viewer
        mock_viewer.is_running.side_effect = [True, False]  # Simulate one loop iteration

        self.env.display_environment()

        mock_launch_passive.assert_called_once_with(mock_model, mock_data)
        mock_mj_step.assert_called_once_with(mock_model, mock_data)
        mock_logger_warning.assert_not_called()

if __name__ == '__main__':
    unittest.main()
