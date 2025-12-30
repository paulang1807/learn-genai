## OpenAI
!!! abstract "Sample Code"
    ```python
    import os
    from dotenv import load_dotenv, find_dotenv
    from openai import OpenAI

    model="gpt-5-nano"
    openai = OpenAI()  # Automatically uses the env variable OPENAI_API_KEY. 

    # We can also specify the api_key parameter explicitly
    _ = load_dotenv(find_dotenv())
    # The following can also be used
    # load_dotenv(override=True)
    api_key  = os.getenv('OPENAI_API_KEY')
    openai = OpenAI(api_key=api_key)

    system_prompt = "You are an useful assistant."
    user_prompt_prefix = "Tell me a joke."

    # Helper function for message
    def get_message(user_prompt):
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]

    # Helper function for prompting
    def get_completion(prompt, model="gpt-3.5-turbo", temperature=0, max_tokens=200):
        response = openai.chat.completions.create(
            model=model,
            messages=get_message(prompt),
            max_tokens=max_tokens,
            temperature=temperature, 
        )
        return response.choices[0].message.content

    prompt = user_prompt_prefix + " It should be about movies."

    response = get_completion(prompt)
    print(response)

    ```

??? info "Using POST"
    ```python
    # The following represents the POST equivalent of the code chunk showin the Usage section above
    import os
    import requests

    api_key  = os.getenv('OPENAI_API_KEY')
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": "gpt-5-nano",
        "messages": [
            {"role": "system", "content": "You are an useful assistant."},
            {"role": "user", "content": "Tell me a joke."}]
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )

    prompt_response = response.json()["choices"][0]["message"]["content"]
    print(prompt_response)
    ```

## Google
!!! abstract "Sample Code"
    ```python
    from openai import OpenAI

    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    google_api_key = os.getenv("GOOGLE_API_KEY")

    gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)

    response = gemini.chat.completions.create(
                model="gemini-2.5-flash-lite", 
                messages=get_message(prompt))

    response.choices[0].message.content
    ```

## Ollama
Ensure that [Ollama is running](../local/#cust-id-ollama-comm) before using this code.
!!! abstract "Sample Code"
    ```python
    # Check if Ollama is running
    # The statement below should return "Ollama is running"
    requests.get("http://localhost:11434").content
    
    OLLAMA_BASE_URL = "http://localhost:11434/v1"

    ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key='dummy')  # api_key just needs a dummy value

    response = ollama.chat.completions.create(
                model="llama3.1:8b", 
                messages=get_message(prompt))

    response.choices[0].message.content
    ```