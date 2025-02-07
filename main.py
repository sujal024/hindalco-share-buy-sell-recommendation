from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session  
from pydantic import BaseModel
from typing import Optional
import os
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi.responses import JSONResponse

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:19731980@db:5432/hindalcodata")

print(f"Using database URL: {SQLALCHEMY_DATABASE_URL}")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Item(Base):
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date)
    price = Column(Float)
    name = Column(String, index=True)
    description = Column(String)

class ItemCreate(BaseModel):
    name: str
    description: Optional[str] = None

class ItemResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None

    class Config:
        from_attributes = True 

class StrategyPerformance(BaseModel):
    total_return: float
    number_of_trades: int
    win_rate: float
    avg_profit_per_trade: float
    max_drawdown: float
    sharpe_ratio: float

Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    print("\n🚀 API Documentation available at: http://localhost:8000/docs")
    print("⚡ Alternative documentation at: http://localhost:8000/redoc\n")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/items/", response_model=ItemResponse)
def create_item(item: ItemCreate, db: Session = Depends(get_db)):
    db_item = Item(**item.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


@app.get("/items/", response_model=list[ItemResponse])
def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    items = db.query(Item).offset(skip).limit(limit).all()
    return items


@app.get("/debug/data")
def debug_data(db: Session = Depends(get_db)):
    try:

        from sqlalchemy import text
        result = db.execute(text("SELECT 1")).fetchone()
        print("Basic connection test successful")

        tables = db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)).fetchall()
        print(f"Available tables: {tables}")


        count = db.execute(text("SELECT COUNT(*) FROM items")).fetchone()
        print(f"Number of records: {count[0] if count else 0}")

        return {
            "status": "success",
            "connection": "working",
            "tables": [t[0] for t in tables],
            "record_count": count[0] if count else 0
        }
        
    except Exception as e:
        print(f"Debug endpoint error: {str(e)}")
        return {
            "status": "error",
            "error_message": str(e),
            "error_type": str(type(e))
        }

@app.post("/debug/sample-data")
def create_sample_data(db: Session = Depends(get_db)):
    try:
        # Create sample data
        start_date = datetime(2023, 1, 1)
        prices = [100.0]  # Starting price
        
        # Generate 100 days of sample price data
        for i in range(99):
            change = np.random.normal(0, 1)  # Random price change
            new_price = prices[-1] * (1 + change/100)
            prices.append(new_price)
        
        # Clear existing data
        db.query(Item).delete()
        
        # Insert new data
        for i, price in enumerate(prices):
            date = start_date + pd.Timedelta(days=i)
            db_item = Item(
                date=date,
                price=float(price),
                name=f"HINDALCO_{i}",
                description=f"Price data for day {i}"
            )
            db.add(db_item)
        
        db.commit()
        return {"message": "Sample data created", "count": len(prices)}
        
    except Exception as e:
        print(f"Error creating sample data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating sample data: {str(e)}"
        )

@app.get("/debug/test-db")
def test_database(db: Session = Depends(get_db)):
    try:
        # Simple connection test
        result = db.execute(text("SELECT 1")).fetchone()
        print("Basic connection test:", result)
        
        return {
            "status": "Connected",
            "test_query": "Success",
            "database_url": SQLALCHEMY_DATABASE_URL.replace(":19731980", ":****")  # Hide password in output
        }
    except Exception as e:
        print(f"Connection error: {str(e)}")
        return {
            "status": "Error",
            "error": str(e),
            "database_url": SQLALCHEMY_DATABASE_URL.replace(":19731980", ":****")
        }




@app.get("/strategy/signals")
def get_trading_signals(
    db: Session = Depends(get_db),
    short_period: int = 20,
    long_period: int = 50
):
    try:
        # Get price data from database
        query = db.query(Item.date, Item.price).order_by(Item.date)
        df = pd.read_sql(query.statement, query.session.bind)
        
        if df.empty:
            return {
                "error": "No data found in database"
            }
        
        # Calculate moving averages
        df['SMA_short'] = df['price'].rolling(window=short_period).mean()
        df['SMA_long'] = df['price'].rolling(window=long_period).mean()
        
        # Generate signals
        df['signal'] = 0  # 0 = hold, 1 = buy, -1 = sell
        df.loc[df['SMA_short'] > df['SMA_long'], 'signal'] = 1
        df.loc[df['SMA_short'] < df['SMA_long'], 'signal'] = -1
        
        # Get signal changes
        signal_changes = df[df['signal'] != df['signal'].shift(1)]
        
        # Format signals
        signals = []
        for _, row in signal_changes.iterrows():
            signals.append({
                "price": float(row['price']),
                "signal": "BUY" if row['signal'] == 1 else "SELL"
            })
        
        return {
            "signals": signals,
            "summary": {
                "total_signals": len(signals),
                "buy_signals": len([s for s in signals if s['signal'] == "BUY"]),
                "sell_signals": len([s for s in signals if s['signal'] == "SELL"]),
                "current_position": "BUY" if df['signal'].iloc[-1] == 1 else "SELL"
            }
        }
        
    except Exception as e:
        print(f"Error generating signals: {str(e)}")
        return {
            "error": f"Error generating signals: {str(e)}",
            "data_shape": len(df) if 'df' in locals() else 0,
            "has_price": "price" in df.columns if 'df' in locals() else False,
            "has_date": "date" in df.columns if 'df' in locals() else False
        }

@app.get("/debug/price-check")
def check_price_data(db: Session = Depends(get_db)):
    try:
        # Get price data
        query = db.query(Item.date, Item.price).order_by(Item.date)
        df = pd.read_sql(query.statement, query.session.bind)
        
        if df.empty:
            return {"error": "No data found"}
        
        # Calculate basic statistics
        stats = {
            "total_records": len(df),
            "date_range": {
                "start": df['date'].dt.strftime('%Y-%m-%d').iloc[0],
                "end": df['date'].dt.strftime('%Y-%m-%d').iloc[-1]
            },
            "price_stats": {
                "min": float(df['price'].min()),
                "max": float(df['price'].max()),
                "mean": float(df['price'].mean()),
                "std": float(df['price'].std())
            }
        }
        
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking price data: {str(e)}"
        )

@app.get("/strategy/recommendation")
def get_recommendation(
    db: Session = Depends(get_db),
    short_period: int = 5,
    long_period: int = 10
):
    try:
        # Get price data
        query = db.query(Item.date, Item.price).order_by(Item.date)
        df = pd.read_sql(query.statement, query.session.bind)
        
        if df.empty:
            return {"error": "No data found"}
            
        # Get latest price
        latest_price = float(df['price'].iloc[-1])
        
        # Calculate moving averages
        df['SMA_short'] = df['price'].rolling(window=short_period).mean()
        df['SMA_long'] = df['price'].rolling(window=long_period).mean()
        
        # Get current signal
        current_signal = "BUY" if df['SMA_short'].iloc[-1] > df['SMA_long'].iloc[-1] else "SELL"
        
        return {
            "current_price": latest_price,
            "recommendation": current_signal,
            "short_ma": float(df['SMA_short'].iloc[-1]),
            "long_ma": float(df['SMA_long'].iloc[-1]),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendation: {str(e)}"
        )