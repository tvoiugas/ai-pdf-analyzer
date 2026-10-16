from datetime import datetime, timedelta, timezone
import os

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import OAuth2PasswordBearer
from pwdlib import PasswordHash
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.sql_models import User

load_dotenv()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
if len(JWT_SECRET_KEY) < 32:
    raise RuntimeError("JWT_SECRET_KEY must be set and at least 32 characters long.")

# Pinned rather than read from the token header, so a forged "alg" can't change how a token is verified.
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "30"))

password_hash = PasswordHash.recommended()

# Checked against when the email is unknown, so a login takes as long whether or not the account exists.
DUMMY_HASH = password_hash.hash("dummy-password")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


async def hash_password(password: str) -> str:
    # Argon2 is deliberately CPU-heavy; keep it off the event loop.
    return await run_in_threadpool(password_hash.hash, password)


async def verify_password(password: str, hashed_password: str) -> bool:
    return await run_in_threadpool(password_hash.verify, password, hashed_password)


def create_access_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    claims = {"sub": str(user_id), "iat": now, "exp": now + timedelta(minutes=JWT_EXPIRE_MINUTES)}
    return jwt.encode(claims, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> int:
    """Return the user id, or raise jwt.InvalidTokenError if the token is forged, expired or malformed."""
    claims = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM], options={"require": ["exp", "sub"]})
    try:
        return int(claims["sub"])
    except ValueError:
        raise jwt.InvalidTokenError("Subject is not a user id.")


async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> User:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        user_id = decode_access_token(token)
    except jwt.InvalidTokenError:
        raise credentials_error

    # Looked up on every request so a deleted account's tokens stop working immediately.
    user = await db.get(User, user_id)
    if user is None:
        raise credentials_error
    return user
