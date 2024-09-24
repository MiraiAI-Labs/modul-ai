import csv
import json
from pathlib import Path

import numpy as np
import uvicorn
from analyst import Analyzer
from cv_analyst import GeminiCVAnalyst
from archetype_chatbot import ArchetypeChatbot
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
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
from typing import Annotated

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
    jobs_analysis_id: int
    job_lists_id: int
    mode: str = "default"

def analyze_task(submission: TextSubmission):
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
        "jobs_analysis_id": submission.jobs_analysis_id,
        "job_lists_id": submission.job_lists_id,
    }
    
    send_webhook("analysis_generated", response)

@app.post("/generate_analysis")
async def analyze(submission: TextSubmission, background_tasks: BackgroundTasks):
    background_tasks.add_task(analyze_task, submission)
    return {"message": "Analysis started"}

def analyze_cv_task(input_text, json_data, review_id: str):
    cv_analyst = GeminiCVAnalyst()
    result = cv_analyst.run_cv_analyst(input_text, json_data)

    response = {
        "result": result,
        "review_id": review_id,
    }
    
    send_webhook("cv_analyzed", response)

@app.post("/analyze_cv")
async def analyze_cv(background_tasks: BackgroundTasks, file: UploadFile, job_analysis: UploadFile, review_id: Annotated[str, Form()]):
    print(f"Received CV file: {file.filename} and job analysis file: {job_analysis.filename} with review_id: {review_id}")
    input_text = await file.read()
    json_data = json.load(job_analysis.file)
    background_tasks.add_task(analyze_cv_task, input_text, json_data, review_id)
    return {"message": "CV analysis started"}

#################################### Bagian ini baru ####################################################

def archetype_quiz_judge_task(text, review_id: str): # Sementara aku set review_id dulu ngikutin atas
    judge = ArchetypeChatbot()
    res = judge.process_text(text)
    
    resp = {
        "result": res,
        "review_id": review_id
    }

    send_webhook("archetype_quiz_result", resp)

@app.post("/archetype_quiz_judge")
async def analyze_cv(background_tasks: BackgroundTasks, input_text : TextSubmission, review_id: Annotated[str, Form()]):
    # print(f"Received CV file: {file.filename} with review_id: {review_id}")
    background_tasks.add_task(archetype_quiz_judge_task, input_text, review_id)
    return {"message": "Archetype Evaluation started"}

#############################################################################################################

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
