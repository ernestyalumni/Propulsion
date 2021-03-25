from flask import Flask
from flask_bootstrap import Bootstrap


# Application factory.
# cf. http://flask.pocoo.org/docs/patterns/appfactories/

def create_app(config_filename=None):
    app = Flask(__name__)

    if not config_filename:
        app.config.from_object(config_filename)

    # Install our Bootstrap extension.
    # After loading, new templates available to derive from in your templates.
    Bootstrap(app)

    # Import blueprints.
    from .views import home


    # Register blueprints.
    app.register_blueprint(home.home_bp)


    return app
