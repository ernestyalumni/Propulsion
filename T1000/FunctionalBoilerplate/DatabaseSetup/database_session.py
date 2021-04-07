from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

"""
The SQLite database file will be in where the function gets run.
"""
engine = create_engine('sqlite:///database.db', echo=True)

db_session = scoped_session(sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine))
