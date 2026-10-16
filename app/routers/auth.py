from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logger import logger
from app.core.security import DUMMY_HASH, create_access_token, hash_password, verify_password
from app.models.sql_models import User


class RegisterRequest(BaseModel):
    email: EmailStr
    # The upper bound caps how much work a single request can make Argon2 do.
    password: str = Field(min_length=8, max_length=128)


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    user = User(email=request.email.lower(), hashed_password=await hash_password(request.password))
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email is already registered.")

    logger.info(f"Registered user {user.id}")
    return {"id": user.id, "email": user.email}


@router.post("/token")
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    # OAuth2's password flow names the field "username"; it carries the email here.
    result = await db.execute(select(User).where(User.email == form.username.lower()))
    user = result.scalars().first()

    password_ok = await verify_password(form.password, user.hashed_password if user else DUMMY_HASH)
    if user is None or not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {"access_token": create_access_token(user.id), "token_type": "bearer"}
