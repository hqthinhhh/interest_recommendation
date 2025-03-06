CREATE TABLE IF NOT EXISTS gender_interest_rules (
    antecedents TEXT NOT NULL,
    consequents TEXT NOT NULL,
    antecedent_support FLOAT NOT NULL,
    consequent_support FLOAT NOT NULL,
    support FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    lift FLOAT NOT NULL
);

CREATE TABLE IF NOT EXISTS age_interest_rules (
    antecedents TEXT NOT NULL,
    consequents TEXT NOT NULL,
    antecedent_support FLOAT NOT NULL,
    consequent_support FLOAT NOT NULL,
    support FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    lift FLOAT NOT NULL
);

CREATE TABLE IF NOT EXISTS age_gender_interest_rules (
    antecedents TEXT NOT NULL,
    consequents TEXT NOT NULL,
    antecedent_support FLOAT NOT NULL,
    consequent_support FLOAT NOT NULL,
    support FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    lift FLOAT NOT NULL
);

CREATE TABLE IF NOT EXISTS user
(
    id     bigserial,
    name   text,
    age    integer,
    gender text
);


