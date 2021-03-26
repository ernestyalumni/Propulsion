from ..DatabaseSetup import Base

from sqlalchemy import Table
from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    String)

"""
@ref Ch. 2, Creating Models with SQLAlchemy, pp. 39, Gaspar and Stouffer (2018)
https://docs.sqlalchemy.org/en/13/orm/basic_relationships.html

@details

Example: say table had following data:

post_id   tag_id
1         1
1         3
2         3
2         4
2         5
3         1
3         2

Many-to-Many adds association table between 2 classes. Association table is
indicated by relationship.secondary argument to relationship()
"""
posts_tags_table = Table(
    'PostTags',
    Base.metadata,
    Column('post_id', Integer, ForeignKey('Posts.pid')),
    Column('tag_id', Integer, ForeignKey('Tags.tid')))


class Tag(Base):
    """
    @ref pp. 38, Many-to-many relationship, Gaspar and Stouffer (2018)

    e.g. Blog posts need multiple tags, so users can easily group similar
    posts. Each tag can refer to many posts, but each post can have multiple
    tags.

    """
    __tablename__ = 'Tags'

    tid = Column(Integer, primary_key=True)
    title = Column(String(255))
    def __init__(self, title):
        self.title = title

    def __repr__(self):
        return "<Tag '{}'>".format(self.title)

