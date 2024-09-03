import csv
import json
from pathlib import Path

import numpy as np
import uvicorn
from analyst import Analyzer
from cv_analyst import GeminiCVAnalyst
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from jobspy import scrape_jobs
from pydantic import BaseModel
import uuid
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
import requests
import hmac
import hashlib

def convert_int64(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError

load_dotenv()

def send_webhook(event, data):
    secret = os.getenv("WEBHOOK_SECRET", "very-long-secret")
    url = os.getenv("WEBHOOK_URL", "http://localhost:8002/webhook")

    payload = {
        "event": event,
        "data": data
    }

    payload_string = json.dumps(payload)
    payload_bytes = json.dumps(payload).encode("utf-8")
    
    secret_bytes = bytes(secret, "utf-8")
    signature = hmac.new( secret_bytes, payload_bytes, hashlib.sha256 )

    headers = {
        "Content-Type": "application/json",
        "Signature": signature.hexdigest(),
    }

    print(f"Sending webhook to {url} with payload {payload_string} and headers {headers}")

    response = requests.post(url, data=payload_string, headers=headers)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

public_dir = Path(__file__).parent / "public"

if not public_dir.exists():
    raise RuntimeError(f"Directory '{public_dir}' does not exist")

app.mount(
    "/public", StaticFiles(directory=public_dir), name="public"
)

class TextSubmission(BaseModel):
    text: str
    mode: str = "default"


@app.post("/generate_analysis")
async def analyze(submission: TextSubmission):
    print(f"Received submission: {submission.text}")
    jobs = scrape_jobs(
        site_name=["indeed", "linkedin", "zip_recruiter"],
        search_term=submission.text,  # Mengakses `text` dari objek submission
        location="indonesia",
        results_wanted=1000,
        hours_old=24 * 30 * 12,
        country_indeed="indonesia",
    )

    print(f"Found {len(jobs)} jobs")

    jobs_file_name = "public/" + str(uuid.uuid4()) + ".csv"
    jobs.to_csv(jobs_file_name, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False)

    print(f"Saved jobs to {jobs_file_name}")

    analyst = Analyzer(csv_dir="./" + jobs_file_name)

    print("Analysing data...")

    analysis_res = {
        "top_job_titles": analyst.top_job_titles(),
        "wordcloud_data": analyst.wordcloud(),
        "top10_job_locs": analyst.top10_job_locations(),
        "job_post_trend": analyst.job_posting_trend(),
        "top10_industries_with_most_jobs": analyst.top10_industries_with_most_jobs(),
        "most_mentioned_skills_and_techstacks": analyst.most_mentioned_skills_and_techstacks(),
        "top10_remote_jobs": analyst.top10_remote_jobs(),
        "top10_non_remote_jobs": analyst.top10_non_remote_jobs(),
        "tech_stacks_overtime": analyst.tech_stacks_overtime(),
    }

    print("Analysis done")

    json_res_name = "public/" + str(uuid.uuid4()) + ".json"

    with open(json_res_name, "w") as json_file:
        json.dump(analysis_res, json_file, indent=4, default=convert_int64)

    print(f"Saved analysis to {json_res_name}")

    response = {
        "jobs_file": jobs_file_name,
        "analysis_file": json_res_name,
        "position": submission.text,
    }
    
    send_webhook("analysis_generated", response)

    return response


# @app.post("/cv_analysis")
# async def cv_analysis(file: UploadFile = File(...)):
#     cv_analyst = GeminiCVAnalyst()
#     with open("./job_analysis.json", "r") as f:
#         json_data = json.load(f)
#     input_text = await file.read()
#     result = cv_analyst.run_cv_analyst(input_text, json_data)
#     return result

@app.post("/analyze_cv")
async def analyze_cv(file: UploadFile = File(...), job_analysis: UploadFile = File(...)):
    cv_analyst = GeminiCVAnalyst()
    input_text = await file.read()
    json_data = json.load(job_analysis.file)
    result = cv_analyst.run_cv_analyst(input_text, json_data)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
