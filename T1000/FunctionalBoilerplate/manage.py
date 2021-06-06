"""
@ref Gaspar and Stouffer (2018), pp. 29, "Creating the user table", pp. 90

@details

Example Usage:

# Tell Flask where to load our shell context
export FLASK_APP=manage.py
flask shell

Run 

create_all_tables()

to create a completely new database if it doesn't exist already, and then add
all the tables deriving from Base, and Base's metadata, if they don't already
exist.

cf. pp. 29, Gaspar and Stouffer (2018), "Creating the user table"

"""
from . import configure_flask_application
from .DatabaseSetup.base import Base
from .DatabaseSetup.create_from_metadata import create_all_tables
from .DatabaseSetup.custom_flask_sqlalchemy import flask_sqlalchemy_db
from .DatabaseSetup.database_session import db_session
from .FunctionalBoilerplate.create_application import create_app
from .FunctionalBoilerplate.example_flask_shell import run_examples
from .Model.comment import Comment
from .Model.post import Post
from .Model.tags import Tag
from .Model.user import User

from flask import url_for
import subprocess

# cf. pp. 44, Gaspar and Stouffer (2018)
# https://flask.palletsprojects.com/en/1.1.x/config/#development-production
app = create_app(configure_flask_application.DevelopmentConfiguration())

@app.shell_context_processor
def make_shell_context():

    """
    @ref https://flask-sqlalchemy.palletsprojects.com/en/2.x/api/?highlight=sqlalchemy#flask_sqlalchemy.SQLAlchemy

    @details This callback used to initialize an application for use with this
    database setup. Never use a database in context of application not
    initialized that way or connections will leak.
    """

    #flask_sqlalchemy_db.init_app(app)

    return dict(
        app=app,
        db=flask_sqlalchemy_db,
        db_session=db_session,
        Base=Base,
        User=User,
        Post=Post,
        Comment=Comment,
        Tag=Tag,
        create_all_tables=create_all_tables,
        run_examples=run_examples)


if __name__ == "__main__":

    shell_context = make_shell_context()