"""
@file static_tree_like_tables.py
"""
from ..DatabaseSetup.base import Base

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship


class AA(Base):

    __tablename__ = "table_name_a_a"

    id = Column(Integer, primary_key=True)
    name = Column(String(250), nullable=False)

    @property
    def serialize(self):
        """Return object data in easily serializable format"""
        return {
            "name": self.name,
            "id": self.id}
