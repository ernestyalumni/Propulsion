from .base import Base
from .database_session import db_session


Base.query = db_session.query_property()