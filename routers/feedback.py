from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import time

from database import get_db
from models import Feedback

router = APIRouter(
    prefix="/feedback",
    tags=["feedback"]
)

class FeedbackRequest(BaseModel):
    user_id: Optional[str] = "anonymous"
    query: str
    response: str
    score: int # +1 or -1
    comment: Optional[str] = None

@router.post("/")
def submit_feedback(feedback: FeedbackRequest, db: Session = Depends(get_db)):
    """
    Submit RLHF feedback (Thumbs Up/Down).
    """
    try:
        db_feedback = Feedback(
            user_id=feedback.user_id,
            query=feedback.query,
            response=feedback.response,
            score=feedback.score,
            comment=feedback.comment,
            timestamp=int(time.time())
        )
        db.add(db_feedback)
        db.commit()
        db.refresh(db_feedback)
        return {"status": "success", "id": db_feedback.id, "message": "Feedback received. Thank you!"}
    except Exception as e:
        print(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))
