import spacy
import requests
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch


def flatten(l):
    return list(set([a for b in l for a in b]))

class Translate():
    subscription_key = '****'
    endpoint = 'https://api.cognitive.microsofttranslator.com'
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
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
    def __init__(self):
        self.step = 0
        self.translator = Translate()
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        self.model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large")
        self.nlp = spacy.load('fr_core_news_lg')

    def get_output_chat(self, input_user):
        input_user_en = self.translator.get_en_from_fr(input_user)
        new_user_input_ids = self.tokenizer.encode(
            input_user_en + self.tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat(
            [chat_history_ids, new_user_input_ids], dim=-1) if self.step > 0 else new_user_input_ids

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

        self.step += 1
        return decoded_outputs_nucleus_fr
    
    def get_words(self, input_user):
        sentences = self.get_output_chat(input_user)
        sents = [nlp(sent) for sent in sentences]
        verbs = flatten([[w.lemma_ for w in sent if w.pos_ == 'VERB'] for sent in sents])
        nouns = flatten([[w.lemma_ for w in sent if w.pos_ == 'NOUN'] for sent in sents])
        pronouns = flatten([[w.lemma_ for w in sent if w.pos_ == 'PRON'] for sent in sents])
        adverbs = flatten([[w.lemma_ for w in sent if w.pos_ == 'ADV'] for sent in sents])
        return {'sentences': sentences, 'verbs': verbs, 'nouns': nouns, 'pronouns': pronouns, 
                'adverbs': adverbs}
        
    def restart_chat(self):
        self.step = 0

def main():
    chat = ChatGeneratorFR()
    chat.restart_chat()
    input_user = "Qu'est ce que tu veux manger?"
    chat.get_words(input_user)
