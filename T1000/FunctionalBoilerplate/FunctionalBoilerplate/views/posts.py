from . import models

from . import (
    Comment,
    Post,
    posts_tags_table,
    Tag,
    User)


from . import flask_sqlalchemy_tables as FSqlTables
from . import db_session
from . import flask_sqlalchemy_db as db

from flask import (
    Blueprint,
    render_template)
from sqlalchemy import func, desc


class PostsBlueprintConstants:
    url_prefix = "/posts"

posts_bp = Blueprint(
    'posts_bp',
    __name__,
    url_prefix=PostsBlueprintConstants.url_prefix)


def sidebar_data():
    recent = Post.query.order_by(Post.publish_date.desc()).limit(5).all()

    top_tags = db_session.query(
        Tag,
        # Using SQLAlchemy func, return a count on a group by query, we're able
        # to order our tags by the most used tags.
        func.count(posts_tags_table.c.post_id).label('total')).join(
            posts_tags_table).group_by(Tag).order_by(
                desc('total')).limit(5).all()

    return recent, top_tags

@posts_bp.route('/')
@posts_bp.route('/<int:page>')
def home(page=1):

    """
    query_object = db.session.query(FSqlTables['Posts']).order_by(
        FSqlTables['Posts'].c.publish_date.desc())
    print(type(query_object))
    print(dir(query_object))
    """
    posts = db.session.query(FSqlTables['Posts']).order_by(
        FSqlTables['Posts'].c.publish_date.desc()).paginate(
            page,
            50,
            False)
    
    print("\n\n Start of posts: \n\n")
    for i in posts.items[:2]:
        print(i)


    recent, top_tags = sidebar_data()

    return render_template(
        'posts/home.html',
        posts=posts,
        recent=recent,
        top_tags=top_tags
        )

@posts_bp.route('/post/<int:post_id>', methods=('GET', 'POST'))
def post(post_id):
    #form = CommentForm()
    return


@posts_bp.route('/posts_by_tag/<string:tag_name>')
def posts_by_tag(tag_name):
    tag = db.session.query(
        FSqlTables['Tags']).filter_by(title=tag_name).first_or_404()
    posts = tag.posts.order_by(FSqlTables['Posts'].publish_date.desc()).all()
    recent, top_tags = sidebar_data()

    return render_template(
        'posts/tag.html',
        tag=tag,
        posts=posts,
        recent=recent,
        top_tags=top_tags)


@posts_bp.route('/posts_by_user/<string::username>')
def posts_by_user(username):
    user = db.session.query(FSqlTables['Users']).filter_by(
        username=username).first_or_404()
    posts = user.posts.order_by(FSqlTables['Posts'].publish_date.desc()).all()
    recent, top_tags = sidebar_data()

    return render_template(
        'posts/user.html',
        user=user,
        posts=posts,
        recent=recent,
        top_tags=top_tags)