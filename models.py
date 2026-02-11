from sqlalchemy import Column, Integer, String
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    age = Column(Integer)

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=True) # Optional user ID
    query = Column(String)
    response = Column(String)
    score = Column(Integer) # +1 for Thumbs Up, -1 for Thumbs Down
    comment = Column(String, nullable=True)
    timestamp = Column(Integer) # Unix timestamp
