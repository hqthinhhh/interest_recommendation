from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List

app = FastAPI()

def categorize_age_group(age):
    if age < 18:
        return "Under_18"
    elif 18 <= age < 25:
        return "18_24"
    elif 25 <= age < 35:
        return "25_34"
    elif 35 <= age < 45:
        return "35_44"
    elif 45 <= age < 55:
        return "45_54"
    elif 55 <= age < 65:
        return "55_64"
    else:
        return "65_plus"

def categorize_age(age):
    if age < 18:
        return "Under 18"
    elif 18 <= age < 25:
        return "18-24"
    elif 25 <= age < 35:
        return "25-34"
    elif 35 <= age < 45:
        return "35-44"
    elif 45 <= age < 55:
        return "45-54"
    elif 55 <= age < 65:
        return "55-64"
    else:
        return "65+"

class Database:
    def __init__(self):
        HOST = 'localhost'
        DATABASE = 'postgres'
        USER = 'postgres'
        PASSWORD = 'postgres'
        self.conn = psycopg2.connect(
            dbname=DATABASE,
            user=USER,
            password=PASSWORD,
            host=HOST,
            port='5432'
        )
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)

    def get_list_user(self):
        self.cursor.execute("SELECT * FROM public.user")
        result = self.cursor.fetchall()
        return result

    def get_recommendation_items(self, payload):
        user = self.cursor.execute("SELECT * FROM public.user where id = %s", (payload["user_id"]))
        user = self.cursor.fetchone()
        gender = user["gender"].lower()
        age = categorize_age(user["age"])
        age_group = categorize_age_group(user["age"])
        age_group_gender = f"{age_group}_{gender}"

        self.cursor.execute(
            "SELECT consequents FROM age_interest_rules WHERE antecedents = %s ORDER BY lift DESC LIMIT %s",
            (age, payload["limit"])
        )
        age_interest_rules_rcm = self.cursor.fetchall()
        print("age_interest_rules_rcm", age_interest_rules_rcm)

        self.cursor.execute(
            "SELECT consequents FROM gender_interest_rules WHERE lower(antecedents) = %s ORDER BY lift DESC LIMIT %s",
            (gender, payload["limit"])
        )
        gender_interest_rules_rcm = self.cursor.fetchall()
        print("gender_interest_rules_rcm", gender_interest_rules_rcm)

        self.cursor.execute(
            "SELECT consequents FROM age_gender_interest_rules WHERE lower(antecedents) = %s ORDER BY lift DESC LIMIT %s",
            (age_group_gender, payload["limit"])
        )
        age_gender_interest_rules_rcm = self.cursor.fetchall()
        print("age_gender_interest_rules_rcm", age_gender_interest_rules_rcm)

        return {
            "age_interest_rules_rcm":age_interest_rules_rcm,
            "gender_interest_rules_rcm": gender_interest_rules_rcm,
            "age_gender_interest_rules_rcm": age_gender_interest_rules_rcm
        }

    def insert_user(self, payload):
        self.cursor.execute(
            "INSERT INTO public.user(name, age, gender) VALUES (%s, %s, %s) RETURNING id",
            (payload.get("name"), payload.get("age"), payload.get("gender"))
        )
        user_id = self.cursor.fetchone()["id"]
        self.conn.commit()
        return {"id": user_id}

class RecommendationRequest(BaseModel):
    user_id: str
    limit: int

class UserCreate(BaseModel):
    name: str
    age: int
    gender: str


@app.get("/get_list_user")
async def get_all_items():
    db = Database()
    items = db.get_list_user()
    return {"data": items}

@app.post("/get_recommendation")
async def get_recommendation(payload: RecommendationRequest):
    db = Database()
    payload = payload.model_dump()
    recommendations = db.get_recommendation_items(payload)
    return {"data": recommendations}

@app.post("/user")
async def get_recommendation(payload: UserCreate):
    db = Database()
    user_create = payload.model_dump()
    user = db.insert_user(user_create)
    return {"data": user}

if __name__ == "__main__":
    db = Database()
    # db.create_table()
