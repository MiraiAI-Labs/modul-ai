import csv
import json

import numpy as np
import uvicorn
from analyst import Analyzer
from cv_analyst import GeminiCVAnalyst
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from jobspy import scrape_jobs
from pydantic import BaseModel


def convert_int64(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextSubmission(BaseModel):
    text: str
    mode: str = "default"


@app.post("/analyze")
async def analyze(submission: TextSubmission):
    jobs = scrape_jobs(
        site_name=["indeed", "linkedin", "zip_recruiter"],
        search_term=submission.text,  # Mengakses `text` dari objek submission
        location="indonesia",
        results_wanted=1000,
        hours_old=24 * 30 * 12,
        country_indeed="indonesia",
    )

    print(f"Found {len(jobs)} jobs")
    jobs.to_csv("jobs.csv", quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False)

    analyst = Analyzer(csv_dir="./jobs.csv")

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

    json_res_name = "job_analysis.json"

    with open(json_res_name, "w") as json_file:
        json.dump(analysis_res, json_file, indent=4, default=convert_int64)

    return analysis_res


@app.post("/cv_analysis")
async def cv_analysis(file: UploadFile = File(...)):
    cv_analyst = GeminiCVAnalyst()
    with open("./job_analysis.json", "r") as f:
        json_data = json.load(f)
    input_text = await file.read()
    result = cv_analyst.run_cv_analyst(input_text, json_data)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
