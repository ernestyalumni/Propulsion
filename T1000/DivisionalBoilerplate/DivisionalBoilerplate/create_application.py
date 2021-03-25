from flask import Flask

# Application factory.
# cf. http://flask.pocoo.org/docs/patterns/appfactories/

def create_app(config_filename='configure_flask_app'):
    app = Flask(__name__)
    app.config.from_object(config_filename)

    # Import blueprints.
    from .home import views as home_views



    # Register blueprints.
    app.register_blueprint(home_views.home_bp)

    return app
