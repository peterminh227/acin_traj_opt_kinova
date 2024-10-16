# Mujoco IIWA Cage

## Getting started

This repository contains code for assembling an environment in Mujoco for simulating interactions between IIWA robot arms and a cage structure.

## Installation

### Using Poetry Package Manager

If you have Poetry installed, you can quickly set up the environment by following these steps:

1. **Install Dependencies**: Run the following command in your terminal:

    ```bash
    poetry install
    ```

2. **Activate Virtual Environment**: You can activate the virtual environment using Poetry:

    ```bash
    poetry shell
    ```

    Alternatively, you can activate the environment in your IDE.

3. **Deactivate Environment**: To deactivate the virtual environment, simply run:

    ```bash
    deactivate
    ```
   
4. **Add project dependencies** to add/remove for example reqeusts or custom packages:

    ```bash
    poetry add requests
    poetry remove requests
   
   poetry add package_name --source custom_repo
   
    ```

5. **Generate wheel**: To generate python package to pip install *.tar.gz and publish to pypi:

    ```bash
    poetry build
    ```

## Usage

To use the Mujoco IIWA Cage Environment, you can interact with the `env_assembly.py` module. This module provides a `CustomEnvironment` class that allows you to assemble and interact with the simulation environment.

```python
# Example usage
 # Create an instance of the environment
 env = CustomEnvironment()

 # Load XML files
 xml_file_cage = "environment_parts/cage/cage.xml"
 model_cage = env.get_model_from_xml(xml_file_cage)
 env.add_model_to_arena(model_cage, "cage", pos=[0, 0, 0], quat=None, is_mocap=False)

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

 # Export model with assets
 env.export_with_assets('showcase_xml_export')

 # Display the environment
 env.display_environment()

```

## Roadmap
Additional features that should be implemented:
- [ ] Add robot next to the cage for additional interaction scenarios.

## Project status
This project is still a work in progress. Contributions are welcome!
