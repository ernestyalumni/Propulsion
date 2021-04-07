from flask import Flask
import os


def create_application(config_object=None):
    app = Flask(__name__)

    if not config_object:
        app.config.from_object(config_object)
    else:
        app.config['SECRET_KEY'] = os.urandom(32)

    # Install our Bootstrap extension.
    # After loading, new templates available to derive from in your templates.
    #bootstrap = Bootstrap(app)

    # Import blueprints.


    # Register blueprints.

    return app
