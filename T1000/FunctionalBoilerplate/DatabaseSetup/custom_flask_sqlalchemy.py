from .base import Base
from .database_session import (
    db_session,
    engine)

from flask_sqlalchemy import SQLAlchemy as FlaskSqlalchemySQLAlchemy


class CustomFlaskSqlalchemySQLAlchemy(FlaskSqlalchemySQLAlchemy):
    """
    @ref https://stackoverflow.com/questions/17842160/using-flask-sqlalchemy-without-the-subclassed-declarative-base
    """
    def create_engine(self, bind=engine, app=None):
        """
        @ref https://flask-sqlalchemy.palletsprojects.com/en/2.x/api/?highlight=sqlalchemy#flask_sqlalchemy.SQLAlchemy.create_engine

        Override this method to have final say over how SQLAlchemy engine is
        created.
        """
        return engine


    def make_declarative_base(self, model_class, metadata):
        """
        @ref https://stackoverflow.com/questions/17842160/using-flask-sqlalchemy-without-the-subclassed-declarative-base

        @details You can have Flask-SQLAlchemy expose the declarative base
        instead of the built-in one.

        Originally, 
  
        make_declarative_base(model, metadata=None)
        Creates the declarative base that all models will inherit from.

        Parameters: * model - base model class (or a tuple of base classes) to
        pass to declarative_base(). Or class returned from declarative_base, in
        which case a new base class is not created.
        * metadata - MetaData instance to use, or none to use SQLAlchemy's
        default.

        cf. https://flask-sqlalchemy.palletsprojects.com/en/2.x/api/?highlight=sqlalchemy#flask_sqlalchemy.SQLAlchemy.make_declarative_base
        """
        return Base


flask_sqlalchemy_db = CustomFlaskSqlalchemySQLAlchemy()