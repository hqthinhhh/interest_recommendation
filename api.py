from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List

app = FastAPI()

class Database:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname='recommendation',
            user='postgres',
            password='postgres',
            host='localhost',
            port='5100'
        )
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)

    def create_table(self):
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS rules (
            id SERIAL PRIMARY KEY,
            itemA TEXT NOT NULL,
            itemB TEXT NOT NULL,
            freqAB INT NOT NULL,
            supportAB FLOAT NOT NULL,
            freqA INT NOT NULL,
            supportA FLOAT NOT NULL,
            freqB INT NOT NULL,
            supportB FLOAT NOT NULL,
            confidenceAtoB FLOAT NOT NULL,
            confidenceBtoA FLOAT NOT NULL,
            lift FLOAT NOT NULL
        );
        '''
        self.cursor.execute(create_table_query)
        self.conn.commit()

    def list_items(self):
        self.cursor.execute("SELECT itemA FROM rules LIMIT 30")
        result = self.cursor.fetchall()
        return result

    def get_recommendation_items(self, selected_item: str, no_items: int):
        self.cursor.execute(
            "SELECT itemB FROM rules WHERE itemA = %s ORDER BY lift DESC LIMIT %s",
            (selected_item, no_items)
        )
        result = self.cursor.fetchall()
        return result

class RecommendationRequest(BaseModel):
    selectedItem: str
    noOfItems: int

@app.get("/all_items")
async def get_all_items():
    db = Database()
    items = db.list_items()
    return {"data": items}

@app.post("/get_recommendation")
async def get_recommendation(selectedItem: str = Form(...), noOfItems: int = Form(...)):
    db = Database()
    recommendations = db.get_recommendation_items(selectedItem, noOfItems)
    return {"data": recommendations}

if __name__ == "__main__":
    db = Database()
    # db.create_table()
