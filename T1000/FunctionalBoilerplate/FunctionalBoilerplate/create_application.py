from flask import Flask
from flask_bootstrap import Bootstrap
import os

# Application factory.
# cf. http://flask.pocoo.org/docs/patterns/appfactories/

def create_app(config_object=None):
    app = Flask(__name__)

    if not config_object:
        app.config.from_object(config_object)
    else:
        app.config['SECRET_KEY'] = os.urandom(32)

    # Install our Bootstrap extension.
    # After loading, new templates available to derive from in your templates.
    #bootstrap = Bootstrap(app)

    # Import blueprints.
    from .views import (about, home, forms)


    # Register blueprints.
    app.register_blueprint(about.about_bp)
    app.register_blueprint(forms.form_bp)
    app.register_blueprint(home.home_bp)

    return app
