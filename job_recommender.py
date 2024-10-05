from io import BytesIO
import pandas as pd
import google.generativeai as genai
import PyPDF2
import yaml


def sanitize_text(text: str) -> str:
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")

with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

class JobRecommender:
    def __init__(self, configs=config):
        self.api_key = configs["GEMINI_API_KEY_COLLECTION"]
        self.generation_conf = configs["generation_config"]
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_LOW_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_LOW_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_LOW_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_LOW_AND_ABOVE",
            },
        ]
        self.pointer = 0
        self.used_cols = ['id', 'title', 'company', 'location', 'date_posted', 'job_type', 'description']
    
    def pick_random_key(self):
        if self.api_key:
            pair_api_key = self.api_key[self.pointer]
            self.pointer = (self.pointer + 1) % len(self.api_key)  # Update pointer
            api_key, email_name = pair_api_key
            print(f"Using API Key from -> {email_name}")
            return api_key
        else:
            return "No more API keys available."
    
    def recommend(self, analysis_res, df_jobs):
        df_jobs = pd.read_csv(df_jobs)[self.used_cols].head(7).to_markdown()
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            system_instruction="Anda adalah seorang konsultan yang ahli membandingkan hasil analisis CV seseorang dengan beberapa lowongan kerja. Anda akan menerima data dari beberapa lowongan kerja yang ada, dan tugas Anda adalah memilih 3 lowongan yang paling cocok dengan hasil analisis CV yang didapatkan agar pelaku CV tersebut dapat memiliki peluang tinggi di terima di lowongan kerja yang Anda pilih. Jawaban Anda haruslah dalam bentuk JSON saja dan mengembalikan 3 index (id) pekerjaan yang Anda katakan paling cocok. Misal, {\"job_ids\": [1, 2, 3]}.",
        )
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(
            f"""
            ANALISIS CV RESULT:
            {analysis_res},

            LOWONGAN PEKERJAAN:
            {df_jobs}
            """
        ).text
        
        try:
            response_json = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            raise ValueError("Failed to parse JSON response")

        if isinstance(response_json, list):
            return response_json
        else:
            print(f"Unexpected response format: {response_json}")
            raise ValueError("Response format is not as expected.")
