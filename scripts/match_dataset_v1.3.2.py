# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 00:10:14 2026

@author: choyc
"""

import duckdb

con = duckdb.connect("mimic_v1.2.db")

# Create tables from CSVs
con.execute("""
CREATE OR REPLACE TABLE cxr AS
SELECT * FROM read_csv_auto('mimic-cxr-2.0.0-metadata.csv',
    header = true,
    delim = ',',
    quote = '"',
    all_varchar = true),
;
""")

con.execute("""
CREATE OR REPLACE TABLE ecg AS
SELECT * FROM read_csv_auto('ecg_record_list.csv');
""")

con.execute("""
CREATE  OR REPLACE TABLE cxr_unique AS
SELECT
    subject_id,
    study_id,
    MIN(cxr_date) AS cxr_date,
    MIN(path) AS path
FROM cxr
GROUP BY subject_id, study_id;
""")

#######################################################
#Execute SQL!
df = con.execute("""
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT study_id) AS distinct_studies
FROM cxr_unique;
""").df()
print(df)

#49K rows
df = con.execute("""            
WITH matched AS (
    SELECT
        e.subject_id AS subject_id,
        e.study_id AS ecg_study_id,
        c.study_id AS cxr_study_id,
        e.path AS ecg_path,
        c.path AS cxr_path,
        e.ecg_date,
        c.cxr_date,
        ABS(DATEDIFF('day', e.ecg_date, CAST(c.cxr_date AS DATE))) AS day_diff,
        ROW_NUMBER() OVER (
            PARTITION BY e.subject_id
            ORDER BY ABS(DATEDIFF('day', e.ecg_date, CAST(c.cxr_date AS DATE))) ASC
        ) AS rn
    FROM ecg e
    JOIN cxr_unique c
      ON e.subject_id = c.subject_id
    WHERE ABS(DATEDIFF('day', e.ecg_date, CAST(c.cxr_date AS DATE))) <= 60
)
SELECT *
FROM matched
WHERE rn = 1
                           
""").df()
print(df)
#df.to_csv("matched_patients_v1.csv", index=False)

print(con.execute("""
SELECT COUNT(*) AS total_rows,
       COUNT(DISTINCT study_id) AS distinct_studies
FROM cxr_unique
""").df())
