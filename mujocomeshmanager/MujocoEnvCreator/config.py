"""
Argument Parser
"""
import configargparse
import numpy as np

def parseArgs():
    file_parser = configargparse.ArgumentParser(formatter_class=configargparse.ArgumentDefaultsHelpFormatter)


    #Global parameters
    file_parser.add_argument('--robot', default='iiwa',choices=['staubliTS2-60', 'iiwa', 'iiwa_5DOF', 'staubliTX2-160L', 'TX2-60L'], help="which robot model to use")
    file_parser.add_argument('--environment', default="cage_single_iiwa",choices=['cage_single_iiwa'], help="which task")

    file_parser.add_argument('--logging_level', default="info",choices=['error', 'debug', 'warning', 'info'], help="which task")


    opts, _ = file_parser.parse_known_args()



    return opts