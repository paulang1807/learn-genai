## Prompts

### Completion Prompts

- Suitable for ==**single-turn tasks**== where the model generates a response based on a single input prompt
- Conversation context/history and role seggregation is not essential
- Works well for content generation, summarization, question-answering etc.

### Chat Prompts

- Designed for ==**multi-turn conversations**== with multiple roles (user, assistant, system etc.)
- Maintains conversation context by processing the entire conversation history
- Ideal for chatbot applications and tasks requiring back-and-forth interactions

## OpenAI
!!! info "API Parameters"
    - **Max Tokens**: Determines the maximum number of tokens that can be generated. This parameter helps control the verbosity of the response. The value typically ranges between 0 and 2048, though it can vary depending on the model and context.
    - **Temperature**: Controls the randomness of the output. It influences how creative (less predictive responses) or deterministic (more predictive responses) the responses are. The value ranges between 0 and 2. ==The default value is 0.8.==
    - **Top P (Nucleus Sampling)**: Dictates the variety in responses by only considering the top ‘P’ percent of probable words. It is an alternative to sampling with temperature and controls the diversity of the generated text. The value ranges between 0 and 1. ==The default value is 0.95.==
    - **Frequency Penalty**: Reduces repetition by decreasing the likelihood of frequently used words. It penalizes new tokens based on their existing frequency in the text so far. The value ranges between -2.0 and 2.0. This setting is disabled by default (value 0)
    - **Presence Penalty**: Promotes the introduction of new topics in the conversation. It penalizes new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. The value ranges between -2.0 and 2.0. Positive values encourage diverse ideas and minimize repetition. This setting is disabled by default (value 0)

??? note "Temperature and Top P sample values"

    | Use Case | Temperature | Top P | Description      |
    | :---------- | :---: |:---: |:-------------------------------------------------- |
    | Chatbot     | 0.5 | 0.5 | Generates responses that balance coherence and diversity resulting in a more natural and engaging conversational tone.      | 
    | Code Generation | 0.2  | 0.1 | Generates code that adheres to established patterns and conventions. Code output is more focussed and syntactically correct.      |
    | Code Comment Generation | 0.3 | 0.2 | Generates concise and relevant code comments that adhere to conventions.      | 
    | Creative Writing | 0.7 | 0.8 | Generates creative and diverse text less constrained by patterns.      |
    | Data Analysis Scripting | 0.2 | 0.1 | Generates focussed, efficient and correct analysis scripts.      |
    | Exploratory Code Writing | 0.6 | 0.7 | Generates creative code that considers and explores multiple solutions.      |

*[GPT]: Generative Pre-Trained Transformers