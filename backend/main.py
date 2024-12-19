from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd

DATABASE_URL = "sqlite:///library.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI()

class Books(Base):
    __tablename__ = "books"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    author = Column(String, nullable=True)
    edition = Column(String, nullable=True)
    branch = Column(String, nullable=False)
    semester = Column(Integer, nullable=False)
    category = Column(String, nullable=True)

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    load_datasets()

def load_datasets():
    datasets = {
        "books": "C:/Users/Admin/Downloads/books.csv",
    }
    for table_name, file_path in datasets.items():
        try:
            df = pd.read_csv(file_path)
            df.to_sql(table_name, con=engine, if_exists="replace", index=False)
            print(f"{table_name} dataset loaded successfully!")
        except Exception as e:
            print(f"Error loading {table_name}: {e}")

def get_sem(usn):
    if "22" in usn:
        return 5
    elif "23" in usn:
        return 4
    else:
        raise HTTPException(status_code=400, detail="Invalid USN")

def get_branch(usn):
    if "ad" in usn:
        return "AD"
    elif "cs" in usn:
        return "CSE"
    elif "cb" in usn:
        return "CB"
    else:
        raise HTTPException(status_code=400, detail="Invalid USN")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Library Recommendation API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/recommendations/{usn}")
def get_recommendations(usn: str, db: Session = Depends(get_db)):
    
    semester = get_sem(usn)
    branch = get_branch(usn)

    syllabus_books = db.query(Books).filter(
        Books.branch == branch,
        Books.semester == semester
    ).all()

    if not syllabus_books:
        raise HTTPException(status_code=404, detail="No syllabus books found for this branch and semester")

    return {
        "syllabus_books": [book.title for book in syllabus_books],
    }
