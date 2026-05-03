# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:11:12 2026

@author: choyc
"""

'''
Combine ecg report and cxg report
subject_id | ecg_study_id | cxr_study_id | ecg_report | cxr_report
'''
import duckdb
import pandas as pd

con = duckdb.connect("report_processing.db")

# Create tables from CSVs
con.execute("""
CREATE OR REPLACE TABLE ecg_reports AS
SELECT * FROM read_csv_auto('ecg_reports.csv',
    header = true,
    delim = ',',
    quote = '"',
    all_varchar = true),
;
""")

con.execute("""
CREATE OR REPLACE TABLE cxr_reports AS
SELECT * FROM read_csv_auto('cxr_reports.csv',
    header = true,
    delim = ',',
    quote = '"',
    all_varchar = true),
;
""")
##################################################
##Combine
df = con.execute("""
SELECT 
    c.subject_id AS subject_id, 
    e.study_id AS ecg_study_id,
    c.cxr_study_id AS cxr_study_id,
    e.merged_ecg_report AS ecg_reports,
    c.cxr_report AS cxr_reports
FROM cxr_reports c
JOIN ecg_reports e
ON c.subject_id = e.subject_id;
""").df()
print(df)

##Output to csv
df.to_csv("combined_reports.csv", index=False)