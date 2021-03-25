from .base import Base
from .database_session import engine


def create_all_tables(input_engine = engine):
    Base.metadata.create_all(bind=input_engine)


if __name__ == "__main__":

    create_all_tables()