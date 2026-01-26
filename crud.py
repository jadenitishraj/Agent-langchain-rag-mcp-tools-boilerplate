from sqlalchemy.orm import Session
import models, schemas

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_users(db: Session, skip: int = 0, limit: int = 10):
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(name=user.name, age=user.age)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def create_users_bulk(db: Session, users: list[schemas.UserCreate]):
    db_users = [models.User(name=u.name, age=u.age) for u in users]
    db.add_all(db_users)
    db.commit()
    for u in db_users:
        db.refresh(u)
    return db_users

def update_user(db: Session, user_id: int, user_update: schemas.UserCreate):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user:
        user.name = user_update.name
        user.age = user_update.age
        db.commit()
        db.refresh(user)
    return user

def delete_user(db: Session, user_id: int):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user:
        db.delete(user)
        db.commit()
    return user
