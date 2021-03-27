from . import Post

from flask import (
    Blueprint,
    render_template)


posts_bp_url_prefix = '/posts'

posts_bp = Blueprint(
    # name 
    'posts_bp',
    # import_name, name of blueprint package, usually __name__. Helps locate
    # root_path for blueprint.
    __name__,
    # Tell Flask that blueprint has its own template and static directories.

    # folder with static files, path is relative to blueprint's root path
    static_folder='static',
    # cf. https://stackoverflow.com/questions/22152840/flask-blueprint-static-directory-does-not-work
    static_url_path='../static'
    template_folder='templates'
    )

def sidebar_data():
	recent = 


@posts_bp.route('/')
def home(page=1):
	posts = Post.query.order_by(Post.publish_date.desc()).paginate(
		page, app.config.get('POSTS_PER_PAGE', 10), False)

	recent, top_tags = sidebar_data()