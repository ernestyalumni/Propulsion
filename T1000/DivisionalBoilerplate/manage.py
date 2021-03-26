from . import configure_flask_application
from .DatabaseSetup.base import Base
from .DatabaseSetup.create_from_metadata import create_all_tables
from .DatabaseSetup.database_session import db_session
from .DivisionalBoilerplate.create_application import create_app
from .Model.comment import Comment
from .Model.post import Post
from .Model.tag import Tag
from .Model.user import User

# cf. pp. 44, Gaspar and Stouffer (2018)
# https://flask.palletsprojects.com/en/1.1.x/config/#development-production
app = create_app(configure_flask_application.DevelopmentConfiguration())


@app.shell_context_processor
def make_shell_context():
    return dict(
        app=app,
        db_session=db_session,
        Base=Base,
        User=User,
        Post=Post,
        Comment=Comment,
        Tag=Tag,
        create_all_tables=create_all_tables)


if __name__ == "__main__":

    shell_context = make_shell_context()