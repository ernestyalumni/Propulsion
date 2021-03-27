from . import Post
from .. import (db_session, Tag, posts_tags_table)

from sqlalchemy import desc, func


def sidebar_data():
    recent = Post.query.order_by(Post.publish_date.desc()).limit(5).all()
    top_tags = db_session.query(
        Tag,
        func.count(posts_tags_table.c.post_id).label('total')).join(
            posts_tags_table).group_by(Tag).order_by(
                desc('total')).limit(5).all()

    return recent, top_tags