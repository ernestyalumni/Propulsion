from flask import Flask

# Application factory.
# cf. http://flask.pocoo.org/docs/patterns/appfactories/

def create_app(config_filename='configure_flask_app'):
    app = Flask(__name__)
    app.config.from_object(config_filename)

    # Import blueprints.


    # Register blueprints.

    return app
