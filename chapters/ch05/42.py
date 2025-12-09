import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
gemini_flash = genai.GenerativeModel("gemini-2.5-flash")
current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../data/ch05/computer_security.csv"
data_csv_path = os.path.join(current_dir, relative_path)
df = pd.read_csv(data_csv_path, header=None, names=["Q", "A", "B", "C", "D", "ANS"]) 


prompt = "問題とAからDの選択肢が与えられるので、問題に全て解答してください。答えの選択肢のみを改行して出力してください。\n\n"

for i, row in df.iterrows():
    tmp_prompt = f"""Q{i+1}.{row["Q"]}\n\nA.{row["A"]}\nB.{row["B"]}\nC.{row["C"]}\nD.{row["D"]}\n\n"""
    prompt = prompt + tmp_prompt

response = gemini_flash.generate_content(prompt)
response = response.text.split("\n")

correct = 0
all = len(df)
for i, res in enumerate(response):
    if df["ANS"][i] == res:
        correct += 1

print(f"正解率：{correct/all}")
