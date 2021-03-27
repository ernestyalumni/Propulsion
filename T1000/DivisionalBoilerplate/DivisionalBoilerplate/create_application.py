from flask import Flask

import os


# Application factory.
# cf. http://flask.pocoo.org/docs/patterns/appfactories/

def create_app(config_object=None):
    app = Flask(__name__)

    if not config_object:
        app.config.from_object(config_object)
    else:
        app.config['SECRET_KEY'] = os.urandom(32)


    # Import blueprints.
    from .home import views as home_views
    from .posts import views as posts_views


    # Register blueprints.
    app.register_blueprint(home_views.home_bp)
    app.register_blueprint(posts_views.posts_bp, url_prefix="/posts")

    return app
