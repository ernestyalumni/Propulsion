class Configuration:
    """Configuration from environment variables."""

    # Static Assets
    STATIC_FOLDER = "static"
    TEMPLATES_FOLDER="templates"

    # Secret key for session management.
    SECRET_KEY = 'my precious'


class DevelopmentConfiguration(Configuration):

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

    SQLALCHEMY_TRACK_MODIFICATIONS = True

    SQLALCHEMY_DATABASE_URI = "sqlite:///database.db"


class ProductionConfiguration(Configuration):

    SQLALCHEMY_TRACK_MODIFICATIONS = False
