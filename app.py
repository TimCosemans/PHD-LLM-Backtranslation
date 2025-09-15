import sys
sys.path.append('./src')  # Adjust the path to your src directory

import streamlit as st
import pandas as pd
import plotly.express as px
from src.backtranslation import backtranslate
from ollama import Client

def main():
    st.title("Backtranslation App")

    # LLM model selection
    L1_llm_model = st.selectbox("Select L1 LLM Model:", ["llama3.2:1b", "moondream:1.8b", "orca-mini:3b"])
    L2_llm_model = st.selectbox("Select L2 LLM Model:", ["llama3.2:1b", "moondream:1.8b", "orca-mini:3b"])
    L3_llm_model = st.selectbox("Select L3 LLM Model:", ["llama3.2:1b", "moondream:1.8b", "orca-mini:3b"])

    # User inputs
    string = st.text_area("Enter the string to be backtranslated:")
    expertise = st.text_input("Enter expertise:")
    guidelines = st.text_input("Enter initial guidelines:")
    original_language = st.text_input("Enter original language:")
    target_language = st.text_input("Enter target language:")

    # Ollama client initialization
    st.session_state.ollama = Client(host='http://ollama:11434')

    if st.button("Backtranslate"):
        translation, results = backtranslate(
            string,
            st.session_state.ollama,
            L1_llm_model,
            L2_llm_model,
            L3_llm_model,
            expertise,
            guidelines,
            original_language,
            target_language
        )

        st.subheader("Translation Result:")
        st.write(translation)

        # Plotting cosine similarity vs iteration using Plotly
        fig = px.line(
            results,
            x='iteration',
            y='cosine_similarity',
            title='Cosine Similarity vs Iteration',
            labels={'iteration': 'Iteration', 'cosine_similarity': 'Cosine Similarity'},
            markers=True
        )
        fig.update_layout(yaxis=dict(range=[0, 1]), xaxis_title="Iteration", yaxis_title="Cosine Similarity")
        st.plotly_chart(fig)

        # Provide download link for the results DataFrame
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="backtranslation_results.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
