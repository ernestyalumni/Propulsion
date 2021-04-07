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