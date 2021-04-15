from ..forms.comment_form import CommentForm
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

import datetime


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
    comment_form = CommentForm()
    if comment_form.validate_on_submit():
        new_comment = Comment()
        new_comment.name = comment_form.name.data
        new_comment.text = comment_form.text.data
        new_comment.post_id = post_id
        new_comment.date = datetime.datetime.now()

        db_session.add(new_comment)
        db_session.commit()

    #post = db.session.query(FSqlTables['Posts']).get_or_404(post_id)
    post = Post.query.get(post_id)
    tags = post.tags
    #comments = post.comments.order_by(Comment.date.desc()).all()

    try:
        comments = post.comments.query.order_by(Comment.date.desc()).all()
    except AttributeError as err:
        comments = []

    recent, top_tags = sidebar_data()

    return render_template(
        'posts/post.html',
        post=post,
        tags=tags,
        comments=comments,
        recent=recent,
        top_tags=top_tags,
        form=comment_form)


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


@posts_bp.route('/posts_by_user/<string:username>')
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