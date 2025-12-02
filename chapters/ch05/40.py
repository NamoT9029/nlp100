import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
gemini_flash = genai.GenerativeModel("gemini-2.5-flash")

prompt =    """
            9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

            ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
            イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
            ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
            """
response = gemini_flash.generate_content(prompt)
print(response.text)