import json
import re

import google.generativeai as genai
import yaml

with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)


class ArchetypeChatbot:
    def __init__(self, configs=config):
        self.api_key = configs["GEMINI_API_KEY_COLLECTION"]
        self.gen_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        self.pointer = 0

    def pick_random_key(self):
        if self.api_key:
            pair_api_key = self.api_key[self.pointer]
            self.pointer = (self.pointer + 1) % len(self.api_key)  # Update pointer
            api_key, email_name = pair_api_key
            print(f"Using API Key from -> {email_name}")
            return api_key
        else:
            return "No more API keys available."

    def process_text(self, input_text):
        api_key = self.pick_random_key()
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=self.gen_config,
            system_instruction=(
                "Cek apakah untuk data ini, jawaban user sudah sesuai dengan kunci jawaban di setiap soalnya. "
                "Beri juga persentase kemiripan atau keterkaitan (nilai anda) jawaban user terhadap kunci jawabannya. "
                "Apabila persentase di bawah 70% DAN jawabannya tidak sesuai menurut Anda (dengan logis tentu saja), "
                "beri penjelasan atau jawaban yang seharusnya. Respon anda harus dalam bentuk JSON array:\n\n"
                '[{"id": ..., "Soal": ..., "Nilai": ..., "Komentar":...},\n'
                '{"id": ..., "Soal": ..., "Nilai": ..., "Komentar":...}, ...]'
                "\n\nGunakan bahasa yang dapat meng-encourage user, terkadang beri semangat kepada user agar tetap bersemangat dalam "
                "meningkatkan kemampuan dirinya. Anda tidak diperbolehkan menjawab hal diluar konteks ini, pastikan supaya jawaban "
                "anda hanya dalam bentuk JSON array."
            ),
        )

        chat_session = model.start_chat(history=[])

        response = chat_session.send_message(input_text)

        cleaned_response = (
            response.text.replace("```json", "").replace("```", "").strip()
        )

        cleaned_response = re.sub(r'(?<!\\)"', r"\"", cleaned_response)

        print(f"Cleaned response: {cleaned_response}")

        try:
            response_json = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            raise ValueError("Failed to parse JSON response")

        if isinstance(response_json, list):
            return response_json
        else:
            print(f"Unexpected response format: {response_json}")
            raise ValueError("Response format is not as expected.")
