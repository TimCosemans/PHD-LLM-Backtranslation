# PHD-LLM-Backtranslation
Backtranslation of surveys using LLMs

Inspired by recent work (Chung & Kim, 2025), we execute backtranslation by using Large Language Models (LLMs). More specifically, we ask an LLM (L1) to translate the original (OR) into Dutch (TR). 

We then ask a different LLM (L2) to perform the backtranslation (BT). These are both instructed to act like bilingual translators with subject matter expertise. 

A third LLM (L3) is used to encode the original and the backtranslation to calculate the cosine similarity. If the cosine similarity is not one (perfect similarity), we instruct this LLM to provide feedback to the translator (L1) to make adaptations to the translations (TR). The process is repeated until the cosine similarity is one (Klotz et al., 2023).  

References

- Chung, J.-B., & Kim, T. (2025). Leveraging large language models for enhanced back-translation: Techniques and applications. IEEE Access.
- Klotz, A. C., Swider, B. W., & Kwon, S. H. (2023). Back-translation practices in organizational research: Avoiding loss in translation. Journal of Applied Psychology, 108(5), 699.

# Getting Started
To get started, you need to have Docker installed on your machine. Once you have Docker installed, you can run the following command in the terminal to start the application:

```bash
docker-compose up --build
```

This command will build the Docker images and start the containers defined in the `docker-compose.yaml` file. It will take a while to download the images and set up the containers, so please be patient. *Everything should work if run on macOSX.*

Go to localhost:8501 to access the Streamlit app. Here, you can perform the backtranslation. 