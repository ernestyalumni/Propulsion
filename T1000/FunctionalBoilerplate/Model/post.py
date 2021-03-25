from ..DatabaseSetup import Base
from .tags import posts_tags_table

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    # A variably sized string type
    Text)
from sqlalchemy.orm import (backref, relationship)

class Post(Base):
    """
    @ref pp. 35, Gaspar and Stouffer (2018) 
    https://docs.sqlalchemy.org/en/13/orm/basic_relationships.html

    @details One-to-many relationship

    One-to-many relationship places foreign key on child table, referencing the
    parent.
    """

    __tablename__ = 'Posts'

    pid = Column(Integer, primary_key=True)

    # A Post must always have a title.
    title = Column(String(255), nullable=False)
    text = Column(Text)
    publish_date = Column(DateTime)

    """
    @ref https://docs.sqlalchemy.org/en/13/core/constraints.html?highlight=foreignkey#sqlalchemy.schema.ForeignKey
    class sqlalchemy.schema.ForeignKey(column,_constraint=None,name=None)

    Parameters
    * column - A single target column for the key relationship. A Column object
    or a column name as a string:
    tablename.columnkey or schema.tablename.columnkey.columnkey is the key
    which has been assigned to the column (defaults to the column name itself)    
    """
    user_id = Column(Integer, ForeignKey('Users.uid'))

    # One to many relationship, but bidirectional relationship.
    comments = relationship("Comment", back_populates="post")

    # Many to many relationship, association table indicated by 
    # relationship.secondary argument to relationship().
    tags = relationship(
        "Tag",
        secondary=posts_tags_table,
        # https://docs.sqlalchemy.org/en/14/orm/backref.html
        # It establishes a collection of Tag objects on Post called Post.tags.
        # It also establishes a .Posts attribute on Tag which will refer to
        # parent Post object.
        backref=backref("post", lazy="dynamic"))

    def __init__(self, title):
        self.title = title

    def __repr__(self):
        return "<Post '{}'>".format(self.title)

