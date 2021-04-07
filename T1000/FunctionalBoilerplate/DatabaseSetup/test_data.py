try:
    from ..Model.post import Post
    from ..Model.tags import Tag
    from ..Model.user import User
except (ImportError, ValueError) as err:
    try:
        from Model.post import Post
        from Model.tags import Tag
        from Model.user import User
    except ImportError as err:
        print("Fail to import: %s", err)
        raise ImportError(err)


try:
    from .database_session import db_session
except (ImportError, ValueError) as err:
    try:
        from DatabaseSetup.database_session import db_session
    except ImportError as err:
        print("Fail to import: %s", err)
        raise ImportError(err)


from faker import Faker
import logging
import random


logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logging.getLogger().setLevel(logging.DEBUG)

log = logging.getLogger(__name__)
faker = Faker()


def generate_tags(n, db = None):
    tags = list()

    def create_and_add_tag(db = None):

        tag = Tag()
        tag.title = faker.color_name()

        if not db:

            try:
                db_session.add(tag)
                db_session.commit()
                tags.append(tag)
            except Exception as err:
                log.error("Fail to add tag %s: %s" % (str(tag), err))
                db_session.rollback()

        else:
            try:
                db.session.add(tag)
                db.session.commit()
                tags.append(tag)
            except Exception as err:
                log.error("Fail to add tag %s: %s" % (str(tag), err))
                db.session.rollback()

    for i in range(n):
        create_and_add_tag(db)

    return tags


def generate_users(n, db = None):

    def create_and_add_user(db = None):
        user = User()
        user.username = faker.name()
        user.password = "password"

        if not db:

            try:
                db_session.add(user)
                db_session.commit()
            except Exception as err:
                log.error("fail to add user %s: %s" % (str(user), err))
                db_session.rollback()

        else:

            try:
                db.session.add(user)
                db.session.commit()
            except Exception as err:
                log.error("Fail to add user: %s: %s" % (str(user), err))
                db.session.rollback()
        return user

    return [create_and_add_user(db) for i in range(n)]


def generate_posts(n, users, tags, db = None):

    def create_and_add_post(db = None):
        post = Post()
        post.title = faker.sentence()
        post.text = faker.text(max_nb_chars=1000)
        post.publish_dat = faker.date_this_century(
            before_today=True,
            after_today=False)

        post.user_id = users[random.randrange(0, len(users))].uid
        post.tags = [tags[random.randrange(0, len(tags))] for i in range(0, 2)]

        try:
            if not db:
                db_session.add(post)
                db_session.commit()
            else:
                db.session.add(post)
                db.session.commit()
        except Exception as err:
            log.error("Fail to add post %s: %s" % (str(post), err))
            if not db:
                db_session.rollback()
            else:
                db.session.rollback()

    return [create_and_add_post(db) for i in range(n)]
