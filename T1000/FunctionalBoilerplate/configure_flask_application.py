"""Class-based Flask app configuration."""
from os import environ
from pathlib import Path
import os

# os.path.abspath(os.path.dirname(__file__))
base_directory = str(Path(__file__).parent.absolute())


class Configuration:
    """Configuration from environment variables."""

    # cf. https://bootstrap-flask.readthedocs.io/en/stable/basic.html
    BOOTSTRAP_SERVE_LOCAL=True

    # Static Assets
    STATIC_FOLDER = "static"
    TEMPLATES_FOLDER="templates"

    # Secret key for session management.
    SECRET_KEY = 'my precious'


class DevelopmentConfiguration(Configuration):

    BOOTSTRAP_BOOTSWATCH_THEME='superhero'

    #FLASK_ENV = environ.get("FLASK_ENV")
    FLASK_ENV="development"
    FLASK_APP = "wsgi.py"

    # Enable debug mode.
    DEBUG = True

    # URI, databasetype+driver://user:password@host:port/db_name
    # SQLite connection string/uri is a path to the database file - relative or
    # absolute.
    # sqlite:///database.db
    # Postgres
    # postgresql+psycopg2://user::password@ip:port/db_name

    SQLALCHEMY_DATABASE_URI = "sqlite:///database.db"


class ProductionConfiguration(Configuration):
    pass