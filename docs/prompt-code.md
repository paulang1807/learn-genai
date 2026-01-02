## [OpenAI](https://colab.research.google.com/drive/1aCKnhpmU3y2btDcp9V7jVq0GM8hTe-2d#scrollTo=bdf93334)
!!! abstract "Sample Code"
    ```python
    import os
    from dotenv import load_dotenv, find_dotenv
    from IPython.display import Markdown, display, update_display
    from openai import OpenAI

    model="gpt-5-nano"
    openai = OpenAI()  # Automatically uses the env variable OPENAI_API_KEY. 
    response_format={"type": "json_object"}

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
    def get_completion(prompt, response_format=None, model="gpt-3.5-turbo", temperature=1, max_tokens=200):
        response = openai.chat.completions.create(
            model=model,
            messages=get_message(prompt),
            max_tokens=max_tokens,
            temperature=temperature, 
            response_format=response_format
        )
        return response.choices[0].message.content

    prompt = user_prompt_prefix + " It should be about movies."

    response = get_completion(prompt)
    print(response)
    ```

!!! example "Streaming Output"
    ```python
    # Helper function for prompting with streaming output
    def get_streaming_completion(prompt, model="gpt-3.5-turbo", temperature=1):
        stream = openai.chat.completions.create(
            model=model,
            messages=get_message(prompt),
            temperature=temperature, 
            stream=True
        )

        response = ""

        # Create an empty markdown handle for streaming output
        display_handle = display(Markdown(""), display_id=True)

        for chunk in stream:
            response += chunk.choices[0].delta.content or ''
            update_display(Markdown(response), display_id=display_handle.display_id)

    get_streaming_completion(prompt)
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

## [Google](https://colab.research.google.com/drive/1aCKnhpmU3y2btDcp9V7jVq0GM8hTe-2d#scrollTo=ml6va219X85-)
!!! abstract "Sample Code"
    ```python
    # Using OpenAI API wrapper
    from openai import OpenAI

    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    google_api_key = os.getenv("GOOGLE_API_KEY")

    gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)

    response = gemini.chat.completions.create(
                model="gemini-2.5-flash-lite", 
                messages=get_message(prompt))

    response.choices[0].message.content
    ```

    ```python
    # Using Gemini API wrapper
    from google import genai
    from google.genai import types

    gemini = genai.Client(api_key=google_api_key)

    # Helper function for adding prompts and responses to Content
    def get_content_from_messages(message, role):
        return types.Content(
            role=role,
            parts=[types.Part.from_text(text=message)]
        )

    # Helper function for generating streaming output
    def get_gemini_streaming_completion(message_content, config, model="gemini-2.5-flash-lite"):
        stream = llm.models.generate_content_stream(
            model=model,
            contents=message_content,
            config=config,
        )
            
        response = ""

        # Create an empty markdown handle for streaming output
        display_handle = display(Markdown(""), display_id=True)
            
        for chunk in stream:
            response += chunk.text or ''
            update_display(Markdown(response), display_id=display_handle.display_id)

    # System Prompt
    system_prompt_config = types.GenerateContentConfig(system_instruction=system_prompt)

    prompt_list = []

    # User Prompt
    prompt = user_prompt_prefix + f"""Tell me a joke"""
    user_message = get_content_from_messages(prompt, "user")

    prompt_list.append(user_message) 

    get_gemini_streaming_completion(prompt_list, system_prompt_config, model="gemini-2.5-flash-lite")
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