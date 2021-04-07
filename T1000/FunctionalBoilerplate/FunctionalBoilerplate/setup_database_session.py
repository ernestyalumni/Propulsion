from ..DatabaseSetup.custom_flask_sqlalchemy import flask_sqlalchemy_db
from ..DatabaseSetup.database_session import db_session

def get_flask_sqlalchemy_db_tables():

    # https://flask-sqlalchemy.palletsprojects.com/en/2.x/api/#flask_sqlalchemy.SQLAlchemy.get_tables_for_bind
    # get_tables_for_bind(bind=None)
    # Returns a list of all tables relevant for a bind.
    #flask_sqlalchemy_db.get_tables_for_bind()

    # cf. https://flask-sqlalchemy.palletsprojects.com/en/2.x/api/#flask_sqlalchemy.SQLAlchemy.metadata
    return flask_sqlalchemy_db.metadata.tables