import sys
sys.path.append('./src')  # Adjust the path to your src directory

import streamlit as st
import pandas as pd
import plotly.express as px
from src.backtranslation import backtranslate
from ollama import Client
import json

# LLM model selection
L1_llm_model = "gemma3:4b"
L2_llm_model = "phi3.5:3.8b"
L3_llm_model = "deepseek-r1:1.5b"

survey = json.load(open('data/survey.json'))

for item in survey["survey"]:
    Q = item["question_number"]
    string = item["text"]
    print(f"Question Number: {Q}, Text: {string}")

    # User inputs
    expertise = "technology adoption research and survey design"
    guidelines = "Translate 'Smart Signal' in English as 'Slim Signaal' in Dutch and vice versa."
    original_language = "English"
    target_language = "Dutch"

    # Ollama client initialization
    ollama = Client(host='http://localhost:11434')

    results = backtranslate(
        string,
        ollama,
        L1_llm_model,
        L2_llm_model,
        L3_llm_model,
        expertise,
        guidelines,
        original_language,
        target_language
    )

    results.to_csv(f"data/results/results_{str(Q)}.csv", index=False)
