import numpy as np
import re

class Translator():
    """
    A class to handle translation and evaluation using Ollama LLMs.
    """
    
    def __init__(self, ollama_client, llm_model, expertise, guidelines = '', 
                 original_language='English', target_language='Dutch'):
        
        self.ollama_client = ollama_client  
        self.llm_model = llm_model
        self.expertise = expertise
        self.guidelines = guidelines
        self.original_language = original_language
        self.target_language = target_language
        self._setup_ollama_model(self.llm_model)

    def translate(self, string):
        """
        Performs translation of the documents in the specified field to the target language using the specified LLM model.
        """

        translation = self._translate(string)

        return translation
    
    def evaluate(self, original, translation, backtranslation):
        """
        Evaluates the quality of the translation by computing the cosine similarity between the original and translated text.
        Returns none if the cosine similarity is one (i.e., texts are identical).
        Returns a prompt to improve the translation if the cosine similarity is below one.
        """
        original_encoded = self.encode(original)
        backtranslation_encoded = self.encode(backtranslation)

        cosine_similarity = np.array(original_encoded) @ np.array(backtranslation_encoded).T

        if cosine_similarity == 1.0:
            return (None, cosine_similarity)
        else:
            prompt = """
            The following is a translation of a text from {original_language} to {target_language} and back.
            This was done by two highly skilled bilingual translators with expertise in {expertise}.
            They received the following guidelines to follow: {guidelines}. 

            The original text is: {original}
            The translated text from the first translator is: {translation}
            The backtranslated text from the second translator is: {backtranslation}

            Give instructions to the first bilingual translator that translates the original on how to improve the translation based on the original text.
            Provide the instructions in less than 100 words. Do not provide any additional commentary.
            """.strip()

            prompt = prompt.format(
                original=original,
                translation=translation,
                backtranslation=backtranslation,
                original_language=self.original_language,
                target_language=self.target_language,
                expertise=self.expertise,
                guidelines=self.guidelines
            )

            response = self.ollama_client.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 500
                }
            )

        return (self._clean_answer(response['response']), cosine_similarity)

    def _translate(self, string):
        """Generate response using Ollama."""

        prompt_template = """
        You are a highly skilled bilingual translator with expertise in {expertise}.
        You translate text from {original_language} to {target_language}.
        You always ensure that the translation is accurate, contextually appropriate, and maintains the original meaning.

        Translate the following text from {original_language} to {target_language}:
        Text: {string}
        
        {guidelines}

        Provide the output strictly in {target_language} as a concise translation.
        Do not provide any additional commentary. 

        """.strip()
            
        prompt = prompt_template.format(
                string=string,
                expertise=self.expertise,
                guidelines=self.guidelines,
                original_language=self.original_language,
                target_language=self.target_language
            )        

        response = self.ollama_client.generate(
            model=self.llm_model,
            prompt=prompt,
            options={
                'temperature': 0.7,
                'top_p': 0.9,
                'max_tokens': 500
            }
        )

        return self._clean_answer(response['response'])
    
    def _setup_ollama_model(self, model_name):
        """Download and setup the Ollama model if not available."""
        try:
            self.ollama_client.show(model_name)
        except:
            print(f"Downloading {model_name}...")
            self.ollama_client.pull(model_name)

    def encode(self, string):
        """
        Encodes a single document using the specified encoder model.
        """
            
        # Using Ollama to embed the documents
        model = self.ollama_client
        result = model.embed(model=self.llm_model, input=string)['embeddings'][0] 

        return result
    
    def _clean_answer(self, answer):
        # Remove everything between <think> and </think>, including the tags
        # https://ollama.com/blog/thinking
        cleaned_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
        return cleaned_answer

    
    