"""
@file species_list.py
"""
from ..utilities.configure_paths import (DataPaths, NISTChemistryWebbookPaths)

from bs4 import BeautifulSoup
from collections import OrderedDict
import csv, requests

class ScrapeWebpage:
    """
    @class ScrapeWebpage
    """
    @staticmethod
    def species_list_columns():
        page = requests.get(NISTChemistryWebbookPaths.species_list())

        # Use Python's built-in html.parser
        soup = BeautifulSoup(page.text, 'html.parser')
        
        species_list_columns = soup.find(id="main").find('ul').find_all('li')
        species_list_columns = [ele.text for ele in species_list_columns]
        return species_list_columns

class ReadAndClean:
    """
    @class ReadAndClean
    """

    @staticmethod
    def _to_list():
        path_to_raw_data = DataPaths().raw()

        species_txt_path = \
            next(path for path in list(path_to_raw_data.glob('**/*')) \
                if "species" in str(path) and ".txt" in str(path))

        # 'r' and newline='' is recommended practice with Python 3 and open
        with open(species_txt_path, 'r', newline='') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            csv_as_list = list(csv_reader)
        return csv_as_list

    @staticmethod
    def _to_None(species_list):
        return [[None if entry=="N/A" else entry for entry in row] \
            for row in species_list]

    @staticmethod
    def to_cleaned_list():
        return ReadAndClean._to_None(ReadAndClean._to_list())

    @staticmethod
    def to_database_ready_data(columns):
        ready_data = []
        cleaned_data = ReadAndClean.to_cleaned_list()
        for row in cleaned_data:
            ready_data.append(OrderedDict(zip(columns, row)))
        return ready_data



