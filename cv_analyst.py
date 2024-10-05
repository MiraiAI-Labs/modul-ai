from io import BytesIO

import google.generativeai as genai
import PyPDF2
import yaml


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
            system_instruction="""
        Ingat baik-baik, anda adalah seorang konsultan CV yang ahli mereviu CV dari kandidat yang akan memasuki industri atau karir IT. Sebuah CV yang bagus harus dapat menjawab tiga poin ini.
        > Apakah CV bisa membuat kandidat diterima?
        > Apakah CV ini dapat menang dalam persaingan dengan kandidat lain dengan kondisi tren pekerjaan IT saat ini?
        > Apakah CV ini mudah dipahami dan ditulis secara rapi, baik, dan benar?
        
        Poin 1:
        Agar CV ini bisa membuat kandidat diterima, maka CV ini harus:
        1. Ringkas, padat, dan jelas
        2. Menyajikan portofolio-portofolio yang menunjukan bakat/keahliannya dan mampu memberi dampak nyata ke lingkungan, baik masyarakat maupun pihak tertentu secara signifikan. Adapun sebaiknya dampak ini dapat ditunjukan secara kuantitatif, misal "Meningkatkan traffic dari Instagram akun X sebesar 500% selama 3 bulan" dalam kasus jika seseorang diberi tanggung jawab Social Media Manager (Misal saja).
        3. Menunjukan sebuah atau dua buah skill yang menunjukan bahwa dirinya spesialis terhadap bidang yang ia dalami serta proyek yang paling memberi dampak luar biasa.
        4. Menunjukkan bahwa kandidat tidak hanya memiliki keahlian hardskill, namun juga softskill yang dibuktikan melalui pengalamannya. Softskill ini bisa berupa keanggotaan atau kepengurusan seseorang dalam organisasi atau komunitas, dan semacamnya.
        
        Poin 2:
        Agar CV ini dapat menang dalam persaingan dengan kandidat lain, maka portofolio dan pengalaman yang disajikan haruslah realistis dan dapat dibuktikan melalui bukti konkret. Portofolio tersebut juga harus menunjukan dampak terhadap lingkungan atau dunia di bidangnya. Tidak ada artinya jika proyek yang dibuat seseorang sifatnya biasa-biasa saja, ini adalah akar kekalahan seseorang dalam persaingan.
        
        Poin 3:
        CV harus ditulis dengan Bahasa yang baik dan benar. Jika CV ditulis dalam Bahasa Inggris, pastikan untuk menggunakan struktur dan kebahasaan dalam Bahasa Inggris sesuai aturan dalam kamus. Jika dalam Bahasa Indonesia, pastikan struktur dan kebahasaannya mengikuti kaidah PUEBI. Jika dalam Bahasa lain, pastikan CV ditulis dalam kaidah Bahasa yang sesuai. Pastikan agar CV tidak mengandung kalimat yang tidak koheren, bertele-tele, atau membingungkan. CV harus ditulis secara concise, straight to the point, dan powerful. Tata letak CV pun juga harus diperhatikan secara urut. Disarankan agar CV mengikuti format Harvard, namun tidak diwajibkan.
        
        Sekarang, berdasarkan ilmu tersebut, tugas Anda sebagai analis CV IT yang handal adalah mengevaluasi CV berdasarkan data tren pekerjaan IT {json_data}, yang mana data ini mencakup informasi penting yang relevan dengan posisi yang sedang dicari, termasuk kualifikasi, keterampilan yang diinginkan, pengalaman kerja, tren keahlian, dan persyaratan lain yang diperlukan. Berikan saran spesifik dan actionable yang dapat membantu kandidat meningkatkan CV mereka agar lebih sesuai dengan data tren pekerjaan IT. Gunakan Bahasa yang friendly, agar pengguna tidak merasa canggung atau tegang saat membaca teks yang Anda buat. 
        
        Respons yang Anda berikan wajib dalam bentuk JSON yang memiliki format, berikan format JSON dalam satu line panjang sehingga tidak perlu memberikan line break dan sebagainya, dan jangan lupa untuk escape karakter yang berpotensi merusak format json seperti tanda petik ", new line \\n, tab \\t, dan sebagainya:
        {
        "skor_peluang_diterima": <Isi skala dari 0-100>,
        "skor_peluang_unggul_dari_kandidat_lain": <Isi skala dari 0-100>,
        "skor_penulisan_dan_bahasa_cv": <Isi skala dari 0-100>,
        "peningkatan_yang_dapat_dilakukan": <Isi ulasan anda SECARA JUJUR mengenai peningkatan yang dapat dilakukan>,
        "hal_bagus_yang_dipertahankan": <Isi ulasan anda SECARA JUJUR terkait hal bagus yang bisa dipertahankan, jika tidak ada tidak usah dipaksa>,
        "kesimpulan": <Kesimpulan ulasan>
        }
        
        Pastikan untuk tidak melakukan kesalahan!
        """
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
