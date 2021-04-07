from ..Model.comment import Comment
from ..Model.post import Post
from ..Model.tags import (posts_tags_table, Tag)
from ..Model.user import User

"""
Obsoleted - see Model subdirectory

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (
    Column,
    Integer,
    String)


engine = create_engine('sqlite:///database.db', echo=True)
db_session = scoped_session(sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine))

Base = declarative_base()
Base.query = db_session.query_property()

# Set your classes here

class User(Base):
    __tablename__ = 'Users'

    uid = Column(Integer, primary_key=True)
    name = Column(String(120), unique=True)
    email = Column(String(120), unique=True)
    password = Column(String(30))

    def __init__(self, name=None, password=None):
        self.name = name
        self.password=password

    def __repr__(self):

        return "<User '{}'>".format(self.name)


# Create tables.
Base.metadata.create_all(bind=engine)
"""
