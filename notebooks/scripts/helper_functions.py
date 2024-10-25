import numpy as np
import re

import nltk
import gensim
from textblob import TextBlob

class GeneralHelper:
    @staticmethod
    def get_cos_sim(em1, em2):
        """Helper function for calculating cosine similarity between two embeddings"""
        return np.dot(em1, em2)/(np.linalg.norm(em1)*np.linalg.norm(em2))

class NLPHelper:
    @staticmethod
    def preprocess_text(text, remove_not=False, should_join=True):
        """Helper function for preprocessing text"""
        textblob_tokenizer = lambda x: TextBlob(x).words
        
        my_stopwords = nltk.corpus.stopwords.words('english')
        if remove_not:
            my_stopwords.remove('not') # retain the word not, so that we can understand negative sentiments
        word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem  # 'extracts' the root of the words from their variations
        my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

        text = ' '.join(word.lower() for word in textblob_tokenizer(text)) # convert to lowercase
        text = re.sub(r'http\S+', '', text) # remove http links
        text = re.sub(r'bit.ly/\S+', '', text) # rempve bitly links
        text = text.strip('[link]') # remove [links]
        text = re.sub('['+my_punctuation + ']+', ' ', text) # remove punctuation
        text = re.sub('\s+', ' ', text) #remove double spacing
        text = re.sub(r"[^a-zA-Z]+", r" ", text) # keep only normal characters
        text_token_list = [word for word in text.split(' ')
                                if word not in my_stopwords] # remove stopwords
        text_token_list = [word_rooter(word) if '#' not in word else word
                            for word in text_token_list] # apply word rooter
        text = ' '.join(text_token_list) # join back the stemmed words for the review
        if should_join:
            return ' '.join(gensim.utils.simple_preprocess(text))
        else:
            return gensim.utils.simple_preprocess(text)
    
    @staticmethod
    def get_embedding(client, prompt, model="TinyLLM"):
        """Helper function for getting embeddings"""
        em = client.embeddings.create(
            model=model,
            input=prompt
        )
        
        return em.data[0].embedding

class ChatHelper:
    @staticmethod
    def get_completion(client, prompt, model="TinyLLM", temperature=0.3, max_tokens=500, top_p=0.95, f_penalty=0, p_penalty=0):
        """Helper function for prompting"""
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature, 
            top_p=top_p, 
            frequency_penalty=f_penalty, 
            presence_penalty=p_penalty
        )
        return response.choices[0].message.content

class PromptHelper:
    def get_email_prompt(subject, recipient_name, email_body):
        return f"""Write a professional email to {recipient_name}.
        Create an innovative subject line based on '{subject}'. 
        Use the following to create the content for the email: {email_body}"""