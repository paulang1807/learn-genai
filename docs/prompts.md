## Prompts

### Completion Prompts

- Suitable for ==**single-turn tasks**== where the model generates a response based on a single input prompt
- Conversation context/history and role seggregation is not essential
- Works well for content generation, summarization, question-answering etc.

### Chat Prompts

- Designed for ==**multi-turn conversations**== with multiple roles (user, assistant, system etc.)
    - **System prompt** - indicates what task the AI is performing and what tone it should use
    - **User prompt** - the prompt from user that the AI should reply to
- Maintains conversation context by processing the entire conversation history
- Ideal for chatbot applications and tasks requiring back-and-forth interactions

### Prompt Structure
- Break down large prompts into steps
    - Use **delimiters** for organizing the prompt

        ??? tip "Stop Sequences"
            - Can be used to signal the end of an example in ==**few-shot prompting**== 
            - Helps to structure the input data for the model
            - Triple backticks, quotes, double hashtags etc.

    - Break large prompts into "sub-prompts" if needed
- **Context**: Brief introduction or background information
    - Include topic, industry/field, relevant links, keywords, datasets, text chunks etc.
    - Constraints or restrictions
- **Persona**: Role or expertise/profession the prompt should use
    - Use phrases such as `Act as`, `Your role is to`, `Imagine you are` etc.
    - Add relevant details such as field of work, company name, time period, region etc.
- **Goal/Intent**: Define what the prompt should do
    - Describe objective
        - Use ==**action verbs**== such as `Analyze`, `Generate`, `Simplify`, `Summarize` etc.
    - Mention any focus areas using phrases such as `Focus on` etc.

    ??? info "Useful Action Verbs and Adjectives"
        - **Verbs**
            - Analyze, Compare, Convert, Customize, Describe, Evaluate, Explain, Generate, Improve, Optimize, Organize, Proofread, Provide, Revise, Rewrite, Share, Simplify, Translate, Write
        - **Adjectives**
            - Clear, Concise, Conversational, Professional

- **Output**
    - **Audience**: Layman, x year old, type of professional, demographics (age/gender/location) etc.
    - **Tone**: Formal, Business, Casual, Academic, Humorous etc.
        - As a Famous Personality e.g. Shakespeare, Tagore etc.
    - **Other Specs**: 
        - Specify constraints (Number of words, points, lines, paragraphs etc.): 
            - Use no more than x words (lines)
            - Summarize using x bullet points, numbered lists etc.
        - Output format with relevant details/templates: 
            - Table with columns x, y and z
            - Json with the following keys/structure
            - Markdown with structured sections and headings
        - Provide step by step instructions
        - Ask for additional details 
            - examples and/or analogies
            - for and against arguments
- **Clarify**: Ask the prompt to ask clarification questions if needed
    - Ask me questions (one at a time) to (add relevant text based on the ask)
- **Validate**
    - Provide relevant sources and citations

??? tip "Code chunks in Prompts"
    Include code chunks in a prompt by enclosing them in single backticks followed by the name of the language:
    ```
    Some text
    `python
    some python code
    `
    More text
    ```


### Basic Examples
- **Provide context**
    - Here is some information to include in /for reference
- **Summarize**
    - Web Pages: Paste the link of the web page followed by the word `summarize`
    - Write a summary of the book xyz by abc
    - Provide an overview of some event
- **Compare**
    - Compare and contrast the concepts of (or the words ..) a and b
- **Explain/Describe** 
    - Explain the concept of xyz in simple terms
    - Explain the concept of xyz using abc analogy
    - Describe the top x abc
- **Generate**
    - Write a simple recipe for xyz
    - Share x tips for abc
    - Help me brainstorm xyz
- **Translate/Convert**
    - Translate the following from one language to another language
    - Convert the following to xyz format
- **Proofread/Review**
    - Proofread and correct any errors in the following
    - Rephrase/Structure the above for maximum impact
    - Review the above and suggest areas for improvement in terms of tone and engagement
    - Is there something I should add to make this resonate better?
- **Follow up Instructions**: Additional prompts to make changes to initial output
    - Provide x alternate versions for 
    - Make the content more/less xyz
    - Incorporate xyz to make the output more abc
    - Add more details / expand on xyz 
    - Focus on xyz

See [Prompt Samples](../prompt-samples) for more examples