from translator import Translator
import pandas as pd

def backtranslate(string, 
                  ollama_client, 
                  L1_llm_model, 
                  L2_llm_model, 
                  L3_llm_model,
                  expertise,
                  guidelines,
                  original_language,
                  target_language, 
                  n_iterations=5): 
    
    """
    Performs backtranslation of the input string using three LLMs.
    L1 translates the original string to the target language.
    L2 backtranslates the translated string to the original language.
    L3 evaluates the quality of the translation by computing the cosine similarity between the original and backtranslated text.
    If the cosine similarity is below one, L3 provides feedback to L1 to improve the translation.
    The process is repeated until the cosine similarity is one.
    
    Args:
        string (str): The input string to be backtranslated.
        ollama_client (OllamaClient): The Ollama client to use for LLM interactions.
        L1_llm_model (str): The LLM model to use for the first translation.
        L2_llm_model (str): The LLM model to use for the backtranslation.
        L3_llm_model (str): The LLM model to use for the evaluation and feedback.
        expertise (str): The area of expertise of the translators.
        guidelines (str): The initial guidelines for the translators.
        original_language (str): The language of the original string.
        target_language (str): The language to translate the original string to.
        n_iterations (int): The maximum number of iterations to perform.
        
        
    Returns:
        translation (str): The final translated string.
        results (pd.DataFrame): A dataframe containing the cosine similarity and guidelines at each iteration.
    """
    
    L1 = Translator(
        ollama_client=ollama_client,
        llm_model=L1_llm_model,
        expertise=expertise,
        guidelines=guidelines,
        original_language=original_language,
        target_language=target_language
    )

    L2 = Translator(
        ollama_client=ollama_client,
        llm_model=L2_llm_model,
        expertise=expertise,
        guidelines=guidelines,
        original_language=target_language,
        target_language=original_language
    ) # Independent backtranslator

    L3 = Translator(
        ollama_client=ollama_client,
        llm_model=L3_llm_model,
        expertise=expertise,
        guidelines=guidelines,
        original_language=original_language,
        target_language=target_language
    )

    cosine_similarity = 0.0
    i = 0

    results = pd.DataFrame(columns=['iteration', 'cosine_similarity', 'guidelines', 'translation'])

    while cosine_similarity < 1.0 and i < n_iterations:  # Limit to 15 iterations to avoid infinite loops
        translation = L1.translate(string)
        backtranslation = L2.translate(translation)
        prompt, cosine_similarity = L3.evaluate(string, translation, backtranslation)

        new_guidelines = guidelines + "\n" + prompt

        L1.guidelines = new_guidelines  # Update guidelines for L1
        i += 1

        df_dictionary = pd.DataFrame([{'iteration': i, 'cosine_similarity': cosine_similarity, 'guidelines': L1.guidelines, 'translation': translation}])
        results = pd.concat([results, df_dictionary], ignore_index=True)
        
        print(f"Iteration {i}: Cosine Similarity = {cosine_similarity}\n")
        print(f"Guidelines for L1: {L1.guidelines}\n")

    return results