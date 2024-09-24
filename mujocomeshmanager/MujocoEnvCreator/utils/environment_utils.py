import functools
import os

def change_directory(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Save the current working directory
        original_directory = os.getcwd()
        # Change to the directory of the function's file
        function_directory = os.path.dirname(os.path.abspath(func.__code__.co_filename))
        os.chdir(function_directory)
        try:
            # Execute the function
            result = func(*args, **kwargs)
        finally:
            # Change back to the original directory
            os.chdir(original_directory)
        return result
    return wrapper
class ChangeDirectoryMeta(type):
    def __new__(cls, name, bases, dct):
        for attr, value in dct.items():
            if callable(value):
                dct[attr] = change_directory(value)
        return super().__new__(cls, name, bases, dct)

def get_joint_ids(list_joint_names, mj_model):
    list_ids= []
    for list_name in list_joint_names:
        list_ids.append(mj_model.joint(list_name).id)
    return list_ids
def get_actuators_ids(list_actuator_names,mj_model):
    list_ids = []
    for list_name in list_actuator_names:
        list_ids.append(mj_model.actuator(list_name).id)
    return list_ids