from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base

import os
from dotenv import load_dotenv

load_dotenv()

# psycopg 3 drives both engines: sync for the Celery worker and scripts, async for the API.
DATABASE_URL = make_url(os.getenv("DATABASE_URL")).set(drivername="postgresql+psycopg")

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

async_engine = create_async_engine(DATABASE_URL)

AsyncSessionLocal = async_sessionmaker(async_engine, autoflush=False, expire_on_commit=False)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as db:
        yield db
