from . import Post
from .db_query import sidebar_data
from .. import Configuration

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
    static_folder='../static',
    # cf. https://stackoverflow.com/questions/22152840/flask-blueprint-static-directory-does-not-work
    #static_url_path='./static',
    template_folder='templates')


@posts_bp.route('/')
@posts_bp.route('/<int:page>')
def home(page=1):
    number_of_rows_per_page = Configuration.POSTS_PER_PAGE

    posts = Post.query.order_by(Post.publish_date.desc()).limit(
        number_of_rows_per_page).offset(page * number_of_rows_per_page)

    recent, top_tags = sidebar_data()

    return render_template(
        'posts/home.html',
        posts=posts,
        recent=recent,
        top_tags=top_tags)