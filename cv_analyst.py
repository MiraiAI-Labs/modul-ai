from io import BytesIO
import yaml
import google.generativeai as genai
import PyPDF2

def sanitize_text(text: str) -> str:
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")

with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

class GeminiCVAnalyst: 
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
    
    def pick_random_key(self):
        global POINTER
        if self.api_key:
            pair_api_key = self.api_key[POINTER]
            POINTER = (POINTER + 1) % len(self.api_key) # move to the next
            api_key, email_name = pair_api_key
            print(f"Using API Key from -> {email_name}")
            return api_key
        else:
            return "No more API keys available."

    def extract_text_from_pdf(self, pdf_path):
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        text = sanitize_text(text)
        return text
    
    def extract_text_from_pdf_buffer(self, pdf_buffer):
        buffer = BytesIO(pdf_buffer)
        pdf_reader = PyPDF2.PdfReader(buffer)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        text = sanitize_text(text)
        return text

    def process_text(self, input_text, json_data):
        input_text = self.extract_text_from_pdf_buffer(input_text)
        
        api_key = self.pick_random_key()
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            safety_settings=self.safety_settings,
            generation_config=self.generation_conf,
            system_instruction=f'''
        "Anda adalah asisten AI yang ahli dalam mereview CV. Tugas Anda adalah melakukan analisis mendalam terhadap CV yang diberikan dan memberikan umpan balik yang konstruktif dan detail. Data berikut, data X yang berisi : {json_data}, mencakup informasi penting yang relevan dengan posisi yang sedang dicari, termasuk kualifikasi, keterampilan yang diinginkan, pengalaman kerja, tren keahlian, dan persyaratan lain yang diperlukan.
        1. Baca dan analisis CV yang disediakan (dalam format PDF). Fokuskan perhatian Anda pada komponen-komponen penting seperti:
        - Ringkasan atau objektif karir
        - Pengalaman kerja yang relevan
        - Keterampilan teknis dan lunak
        - Pendidikan dan sertifikasi
        - Proyek atau publikasi yang signifikan
        - Tata letak dan struktur CV

        2. Bandingkan isi CV dengan data X untuk menilai kesesuaian antara pengalaman dan keterampilan kandidat dengan kondisi pekerjaan saat ini berdasarkan data tersebut.

        3. Identifikasi kekuatan utama dari CV dan area yang dapat ditingkatkan, seperti:
        - Kesesuaian pengalaman kerja dengan posisi yang diinginkan
        - Kualitas deskripsi pekerjaan sebelumnya dan pencapaian yang tercantum
        - Kelengkapan informasi terkait keterampilan dan kualifikasi
        - Keteraturan dan kerapian presentasi CV

        4. Berikan saran spesifik dan actionable yang dapat membantu kandidat meningkatkan CV mereka agar lebih sesuai dengan data X.

        5. Jika ada kesalahan tata bahasa, ejaan, atau format yang ditemukan, berikan koreksi dan rekomendasi perbaikan.

        Anda dapat memulai dengan membaca CV dan kemudian memberikan analisis serta umpan balik yang komprehensif berdasarkan pedoman di atas."
        ''',
        )

        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(input_text)

        response_text = response.text
        return response_text
    
    def run_cv_analyst(self, text, json_data, MAXIMUM_TRY=10):
        for _ in range(MAXIMUM_TRY):
            try:
                summary = self.process_text(text, json_data)
                return summary
            except Exception as e:
                print(f"error: {e}")
        return "Maaf.. saat ini kami belum bisa melakukan evaluasi CV Anda, mungkin silahkan coba lagi nanti ya ^_^"
