"""
@ref Gaspar and Stouffer (2018), pp. 29, "Creating the user table", pp. 90

@details

Example Usage:

# Tell Flask where to load our shell context
export FLASK_APP=manage.py
flask shell

Run 

create_all_tables()

to create a completely new database if it doesn't exist already, and then add
all the tables deriving from Base, and Base's metadata, if they don't already
exist.

cf. pp. 29, Gaspar and Stouffer (2018), "Creating the user table"

"""
from . import configure_flask_application
from .DatabaseSetup.base import Base
from .DatabaseSetup.create_from_metadata import create_all_tables
from .DatabaseSetup.database_session import db_session
from .FunctionalBoilerplate.create_application import create_app
from .Model.comment import Comment
from .Model.post import Post
from .Model.tags import Tag
from .Model.user import User


# cf. pp. 44, Gaspar and Stouffer (2018)
app = create_app(".configure_flask_application.DevelopmentConfiguration")

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