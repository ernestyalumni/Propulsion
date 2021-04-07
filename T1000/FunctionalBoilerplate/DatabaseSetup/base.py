from sqlalchemy.ext.declarative import declarative_base

from .database_session import db_session


# Base from sqlalchemy.ext.declarative
Base = declarative_base()
Base.query = db_session.query_property()