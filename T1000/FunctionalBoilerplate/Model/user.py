try:
    from ..DatabaseSetup import Base
except (ImportError, ValueError) as err:
    try:
        from DatabaseSetup import Base
    except ImportError as err:
        print("Fail to import: %s", err)
        raise ImportError(err)

        
from sqlalchemy import (
    Column,
    Integer,
    String)
from sqlalchemy.orm import relationship


class User(Base):
    __tablename__ = 'Users'

    uid = Column(Integer, primary_key=True)

    # Improve models by setting constraints on the data.
    # RDBMS indexes are used to improve query performance; be careful as this
    # comes of additional writes on INSERT, UPDATE, and DELETE functions costs,
    # as well as storage increase.
    # Take into account that an index is used to reduce O(N) lookup on certain
    # table columns that may be frequently used, or that in tables with huge
    # number of rows where a linear lookup is simply not possible in
    # production. Index query performance can go from logarithmic to O(1).
    # This is possible at cost of additional writes and checks.
    username = Column(String(255), nullable=False, index=True, unique=True)
    password = Column(String(255))

    def __init__(self, username = None):
        self.username = username

    def __repr__(self):
        return "<User '{}'>".format(self.username)    

    """
    @ref https://docs.sqlalchemy.org/en/13/orm/relationship_api.html#sqlalchemy.orm.relationship
    function sqlalchemy.orm.relationship(argument,secondary=None,foreign_keys=
    None,order_by=False,back_populates=None,lazy='select',single_parent=False,
    active_history=False)

    Parameters
    * argument - A mapped class, or actual Mapper instance, representing target
    of the relationship.
    relationship.argument may also be passed a callable function which is
    evaluated at mapper initialization time, and maybe passed a string name
    when using Declarative.
    * backref - Indicates string name of a property to be pasced on related
    mapper's class that'll handle this relationship in other direction. The
    other property will be created automatically when mappers are configured.
    
    * lazy='select' - specifies how related items should be loaded.
      - dynamic - attribute will return a pre-configured Query object for all
      read operations, onto which further filtering operations can be applied
      before iterating the results
    """
    posts = relationship(
        # string name of class when using Declarative
        'Post',
        # cf. pp. 36, Gaspar, Stouffer (2018)
        # backref parameter gives ability to access and set our User class via
        # Post.user
        backref='user',
        lazy='dynamic')