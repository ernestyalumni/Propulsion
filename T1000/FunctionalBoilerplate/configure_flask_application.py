"""Class-based Flask app configuration."""
from os import environ
from pathlib import Path


# os.path.abspath(os.path.dirname(__file__))
base_directory = str(Path(__file__).parent.absolute())


class Configuration:
    """Configuration from environment variables."""

    # Static Assets
    STATIC_FOLDER = "static"
    TEMPLATES_FOLDER="templates"


class DevelopmentConfiguration(Configuration):

    #FLASK_ENV = environ.get("FLASK_ENV")
    FLASK_ENV="development"
    FLASK_APP = "wsgi.py"

    # Enable debug mode.
    DEBUG = True

    # Secret key for session management.
    SECRET_KEY = 'my precious'

    # URI, databasetype+driver://user:password@host:port/db_name
    # SQLite connection string/uri is a path to the database file - relative or
    # absolute.
    # sqlite:///database.db
    # Postgres
    # postgresql+psycopg2://user::password@ip:port/db_name

    SQLALCHEMY_DATABASE_URI = "sqlite:///database.db"


class ProductionConfiguration(Configuration):
    pass