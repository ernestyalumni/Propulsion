from .. import models
from ..models import (Comment, Post, posts_tags_table, Tag, User)
from ..setup_database_session import (
    db_session,
    flask_sqlalchemy_db,
    get_flask_sqlalchemy_db_tables)


flask_sqlalchemy_tables = get_flask_sqlalchemy_db_tables()