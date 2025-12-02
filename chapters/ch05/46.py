import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
gemini_flash = genai.GenerativeModel("gemini-2.5-flash")

prompt =   """
            以下のお題に対する川柳の案を10個作成してください。
            それぞれの川柳のみを改行して出力してください。
            お題「IT」
            """
response = gemini_flash.generate_content(prompt)
print(response.text)