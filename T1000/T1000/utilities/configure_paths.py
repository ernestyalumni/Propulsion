"""
@file configure.py

@brief Helper functions that wrap Python 3's configparser.
"""
from collections import namedtuple
from pathlib import Path

import configparser

def __setup_paths():
    """
    @fn _setup_paths
    @brief Auxiliary function to set up configure.py, making project aware of
    its file directory paths or "position."
    """
    current_filepath = Path(__file__).resolve() # Resolve to the absolute path.

    # These values are dependent upon where this file, configure.py, is placed.
    number_of_parents_to_project_path = 2
    number_of_parents_to_config_file = 1

    filepath_to_config_ini = \
        current_filepath.parents[number_of_parents_to_config_file] / \
            "config.ini"

    Setup = namedtuple('Setup', \
       ['number_of_parents_to_project_path', \
       'configure_filepath',
       'config_ini_filepath'])

    return Setup(number_of_parents_to_project_path, \
        current_filepath,
        filepath_to_config_ini)

def _config():
    """
    @fn _config
    @brief Returns a ConfigParser instance for the config.ini of this project.
    """
    _, _, filepath_to_config_ini = __setup_paths()
    config = configparser.ConfigParser()
    config.read(str(filepath_to_config_ini))
    return config

def _raw_config():
    """
    @fn _config
    @brief Returns a ConfigParser instance for the config.ini of this project.

    This reference explains why we need this for URLs in config.ini (to deal
    with '//' double backslash characters'):
    @ref https://stackoverflow.com/questions/47640354/reading-special-characters-text-from-ini-file-in-python
    """
    _, _, filepath_to_config_ini = __setup_paths()
    config = configparser.RawConfigParser()
    config.read(str(filepath_to_config_ini))
    return config

def _project_path():
    """
    @fn _project_path
    @brief Returns a pathlib Path instance to this project's file path.
    """
    number_of_parents_to_project_path, current_file_path, _ = __setup_paths()
    return current_file_path.parents[number_of_parents_to_project_path]

class DataPaths:

    data_subdirectory_name = "data"

    def raw(self):
        """
        @fn raw
        """
        raw_subdirectory_name = "raw"
        return \
            (_project_path() / \
                self.data_subdirectory_name / \
                raw_subdirectory_name).resolve()

    def processed(self):
        """
        @fn processed
        """
        processed_subdirectory_name = "processed"
        return \
            (_project_path() / \
                self.data_subdirectory_name / \
                processed_subdirectory_name).resolve()

class NISTChemistryWebbookPaths:
    """
    @fn NISTChemistryWebbookPaths

    @brief Paths for the NIST Chemistry Webbook, saved in config.ini.
    """
    @staticmethod
    def species_list():
        """
        @fn species_list
        """
        return _raw_config()["Paths.NIST_Chemistry_Webbook.Species_List"]["download_page"]
