from ..DatabaseSetup.custom_flask_sqlalchemy import flask_sqlalchemy_db

from flask import Flask
#from flask_bootstrap import Bootstrap
import os

# Application factory.
# cf. http://flask.pocoo.org/docs/patterns/appfactories/

def create_app(config_object=None):
    app = Flask(__name__)

    if config_object != None:
        print("\nConfiguring from config_object\n")
        app.config.from_object(config_object)
        print(app.config['SQLALCHEMY_DATABASE_URI'])
    else:
        print("\n no configuration object \n")
        app.config['SECRET_KEY'] = os.urandom(32)
        app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///database.db"
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    flask_sqlalchemy_db.init_app(app)

    # Install our Bootstrap extension.
    # After loading, new templates available to derive from in your templates.
    #bootstrap = Bootstrap(app)

    # Import blueprints.
    from .views import (about, class_based_views, examples, forms, home, posts)


    # Register blueprints.
    app.register_blueprint(about.about_bp)
    app.register_blueprint(examples.examples_bp)
    app.register_blueprint(forms.form_bp)
    app.register_blueprint(home.home_bp)
    app.register_blueprint(posts.posts_bp)
    app.register_blueprint(class_based_views.class_based_views_bp)

    return app
