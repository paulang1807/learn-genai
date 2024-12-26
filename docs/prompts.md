## Prompts

### Completion Prompts

- Suitable for ==**single-turn tasks**== where the model generates a response based on a single input prompt
- Conversation context/history and role seggregation is not essential
- Works well for content generation, summarization, question-answering etc.

### Chat Prompts

- Designed for ==**multi-turn conversations**== with multiple roles (user, assistant, system etc.)
- Maintains conversation context by processing the entire conversation history
- Ideal for chatbot applications and tasks requiring back-and-forth interactions

### Prompt Structure
- Break down large prompts into steps
    - Use separate prompts if needed
- **Context**: Brief introduction or background information
    - Include topic, industry/field, relevant links, keywords, datasets etc.
    - Constraints or restrictions
- **Persona**: Role or expertise/profession the prompt should use
    - Use phrases such as `Act as`, `Your role is to` etc.
    - Add relevant details such as field of work, company name, time period, region etc.
- **Goal/Intent**: Define what the prompt should do
    - Describe objective
        - Use ==**action verbs**== such as `Analyze`, `Generate`, `Simplify`, `Summarize` etc.
    - Mention any focus areas using phrases such as `Focus on` etc.

    ??? info "Useful Action Verbs"
        Analyze, Compare, Convert, Customize, Describe, Evaluate, Explain, Generate, Improve, Optimize, Organize, Proofread, Provide, Revise, Rewrite, Share, Simplify, Translate, Write,

- **Output**
    - **Audience**: Layman, x year old, type of professional, demographics (age/gender/location) etc.
    - **Tone**: Formal, Business, Academic, Humorous etc.
        - As a Famous Personality e.g. Shakespeare, Tagore etc.
    - **Other Specs**: 
        - Number of words, points, lines, paragraphs etc.: 
            - Use no more than x words (lines)
            - Summarize using x bullet points, numbered lists etc.
        - Output format with relevant details/templates: 
            - Table with columns x, y and z
            - Json with the following keys/structure
            - Markdown with structured sections and headings
        - Provide step by step instructions
- **Validate**
    - Provide relevant sources and citations

### Basic Examples
- **Summarize**
    - Web Pages: Paste the link of the web page followed by the word `summarize`
    - Write a summary of the book xyz by abc
    - Provide an overview of some event
- **Compare**
    - Compare and contrast the concepts of (or the words ..) a and b
- **Explain/Describe** 
    - Explain the concept of xyz in simple terms
    - Describe the top x abc
- **Generate**
    - Write a simple recipe for xyz
    - Share x tips for abc
- **Translate/Convert**
    - Translate the following from one language to another language
    - Convert the following to xyz format
- **Proofread**
    - Proofread and correct any errors in the following
- **Follow up Instructions**: Additional prompts to make changes to initial output
    - Provide x alternate versions for 
    - Make the content more/less xyz
    - Incorporate xyz to make the output more abc
    - Add more details / expand on xyz 
    - Focus on xyz


*[GPT]: Generative Pre-Trained Transformers