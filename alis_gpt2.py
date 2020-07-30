import uuid
import spacy
import requests
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch


def flatten(l):
    return list(set([a for b in l for a in b]))

class Translate():
    def __init__(self, azure_cog_key):
        self.subscription_key = azure_cog_key
        self.endpoint = 'https://api.cognitive.microsofttranslator.com'
        self.headers = {
            'Ocp-Apim-Subscription-Key': self.subscription_key,
            'Ocp-Apim-Subscription-Region': 'westeurope',
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

    def get_en_from_fr(self, entry):
        body = [{
            'text': entry
        }]

        path = '/translate?api-version=3.0'
        params = '&from=fr&to=en'
        constructed_url = self.endpoint + path + params

        request = requests.post(constructed_url, headers=self.headers, json=body)
        response = request.json()
        translation = response[0]['translations'][0]['text']
        return translation

    def get_fr_from_en(self, entry):
        body = [{
            'text': entry
        }]

        path = '/translate?api-version=3.0'
        params = '&from=en&to=fr'
        constructed_url = self.endpoint + path + params

        request = requests.post(constructed_url, headers=self.headers, json=body)
        response = request.json()
        translation = response[0]['translations'][0]['text']
        return translation

class ChatGeneratorFR():
    def __init__(self, azure_cog_key):
        self.translator = Translate(azure_cog_key)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        self.model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large")
        self.nlp = spacy.load('fr_core_news_lg')
        self.chat_history_ids = None

    def get_output_chat(self, input_user):
        input_user_en = self.translator.get_en_from_fr(input_user)
        new_user_input_ids = self.tokenizer.encode(
            input_user_en + self.tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        if self.chat_history_ids is None:
            bot_input_ids = new_user_input_ids
        else:
            bot_input_ids = torch.cat(
                [self.chat_history_ids, new_user_input_ids], dim=-1)

        # generated a response while limiting the total chat history to 1000 tokens, 
        sample_output_nucleus = self.model.generate(
            bot_input_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True, 
            max_length=50, 
            top_p=0.95, 
            top_k=50,
            num_return_sequences=5
        )
        
        decoded_outputs_nucleus = [self.tokenizer.decode(
            res, skip_special_tokens=True) for res in sample_output_nucleus[:, bot_input_ids.shape[-1]:]]
        decoded_outputs_nucleus_fr = [self.translator.get_fr_from_en(res) for res in decoded_outputs_nucleus]

        self.chat_history_ids = sample_output_nucleus

        return decoded_outputs_nucleus_fr
    
    def get_words(self, input_user):
        sentences = self.get_output_chat(input_user)
        sents = [self.nlp(sent) for sent in sentences]
        verbs = flatten([[w.lemma_ for w in sent if w.pos_ == 'VERB'] for sent in sents])
        nouns = flatten([[w.lemma_ for w in sent if w.pos_ == 'NOUN'] for sent in sents])
        pronouns = flatten([[w.lemma_ for w in sent if w.pos_ == 'PRON'] for sent in sents])
        adverbs = flatten([[w.lemma_ for w in sent if w.pos_ == 'ADV'] for sent in sents])
        return {'sentences': sentences, 'verbs': verbs, 'nouns': nouns, 'pronouns': pronouns, 
                'adverbs': adverbs}
        
    def restart_chat(self):
        self.chat_history_ids = None

def main():
    azure_cog_key = '<azure_cognitive_services_key>'
    chat = ChatGeneratorFR(azure_cog_key)
    chat.restart_chat()
    input_user = "Qu'est ce que tu veux manger?"
    chat.get_words(input_user)
