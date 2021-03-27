from ..DatabaseSetup import Base

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text)
from sqlalchemy.orm import relationship
import datetime


class Comment(Base):
    """
    @details One To Many relationship, with bidirectional relationship.
    @ref https://docs.sqlalchemy.org/en/13/orm/basic_relationships.html
    """

    __tablename__ = 'Comments'

    cid = Column(Integer, primary_key=True)
    name = Column(String(255))
    text=Column(Text)
    date = Column(DateTime, default=datetime.datetime.now)
    post_id = Column(Integer, ForeignKey('Posts.pid'))

    post = relationship('Post', back_populates="comments")

    def __repr__(self):
        return "<Comment '{}'>".format(self.text[:15])
