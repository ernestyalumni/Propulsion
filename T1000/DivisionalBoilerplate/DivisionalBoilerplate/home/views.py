from flask import (
    Blueprint,
    render_template)


"""
@ref https://flask.palletsprojects.com/en/1.1.x/api/
@details Blueprint Objects

class flask.Blueprint(name,import_name,static_folder=None,static_url_path=None,
template_folder=None,url_prefix=None,subdomain=None,url_defaults=None,
root_path=None,cli_group=<object object>)
"""

home_bp = Blueprint(
    # name 
    'home_bp',
    # import_name, name of blueprint package, usually __name__. Helps locate
    # root_path for blueprint.
    __name__,
    # Tell Flask that blueprint has its own template and static directories.

    # folder with static files, path is relative to blueprint's root path
    static_folder='./home/static',
    template_folder='./templates'
    )

@home_bp.route('/')
def home():
    return render_template(
        'home/placeholder.home.html',
        title="Functional Boilerplate home",
        subtitle="Demonstrate Functional structure for Flask")