import os
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
gemini_flash = genai.GenerativeModel("gemini-2.5-flash")

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../outputs/ch05/result46.txt"
data_path = os.path.join(current_dir, relative_path)

prompt ="""以下の川柳をそれぞれ1～10の10段階で評価してください。点数が高いほど高評価です。点数のみを改行して出力してください。\n\n"""

with open(data_path) as f:
    texts = f.readlines()

    for i, text in enumerate(texts):
        prompt = prompt + f"{i+1}. {text}"

response = gemini_flash.generate_content(prompt)
response = response.text.split("\n")

for res in response:
    print(res)