"""
@file tree_like.py
"""
from ..utilities.configure_database import sqlite_dialect_absolute_url
from ..utilities.configure_paths import DataPaths

from sqlalchemy import create_engine

class TreeLikeConfiguration:

    @staticmethod
    def _tree_like_database_name():

        return "tree_like.db"


    @staticmethod
    def _tree_like_database_url():

        return (
            sqlite_dialect_absolute_url()
                + str(DataPaths().processed())
                + "/"
                + TreeLikeConfiguration._tree_like_database_name())

    @staticmethod
    def _create_tree_like_engine():
        return create_engine(TreeLikeConfiguration._tree_like_database_url())


    
