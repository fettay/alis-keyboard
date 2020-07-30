import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import word2vec
import nltk
import string


EMBEDDING_DIM = 200
SOS = 0.5 * torch.ones(200)
TOP_K = 10


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(EMBEDDING_DIM, hidden_size)

    def forward(self, input, hidden):
        embedded = input.view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(EMBEDDING_DIM, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = input.view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class NextWordPredictor:

    def __init__(self, encoder_path, decoder_path, word2vec_bin):
        self.encoder = torch.load(encoder_path, map_location=torch.device('cpu'))
        self.decoder = torch.load(decoder_path, map_location=torch.device('cpu'))
        self.embedding_model = word2vec.load(word2vec_bin)
    
    def tokenize(self, sentence):
        tokens = nltk.word_tokenize(sentence, language='french')
        enriched_tokens = []
        for token in tokens:
            if token in string.punctuation:
                continue
            
            enriched_tokens.extend(token.split('-'))
        return enriched_tokens
    
    def format_sentence(self, sentence):
        tokens = self.tokenize(sentence)
        tokens.append('<eos>')
        return tokens

    def embed_sentence(self, words):
        all_embeddings = []
        for word in words:
            try:
                embedding = self.embedding_model.get_vector(word)
            except Exception:
                embedding = np.zeros(200)
            all_embeddings.append(embedding)
        return torch.tensor(all_embeddings)
    
    def generate_input(self, sentence):
        sentence = self.format_sentence(sentence)
        embedded = self.embed_sentence(sentence)
        return embedded.type(torch.FloatTensor)

    def get_vector(self, index):
        if index <= len(self.embedding_model.vectors) - 1:
            return self.embedding_model.vectors[index]
        if index == len(self.embedding_model.vectors):
            return np.ones(200)
        return np.zeros(200)

    def get_word(self, index):
        if index <= len(self.embedding_model.vectors) - 1:
            return self.embedding_model.vocab[index]
        if index == len(self.embedding_model.vectors):
            return "<eos>"
        return "<ukn>"

    def get_index(self, word):
        try:
            return self.embedding_model.ix(word)
        except Exception:
            return len(self.embedding_model.vectors) + 1

    def encode_sentence(self, sentence, max_length=50):
        if sentence is None:
            sentence = ""
        sentence = sentence.lower()
        with torch.no_grad():
            input_tensor = self.generate_input(sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei],
                                                    encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        return encoder_hidden

    def predict_next_word(self, hidden_state, prev_word=None):
        if prev_word is not None:
            index = self.get_index(prev_word)
            decoder_input = torch.tensor(self.get_vector(index)).type(torch.FloatTensor)
        else:
            decoder_input = SOS

        decoder_output, decoder_hidden = self.decoder(decoder_input, hidden_state)
        _, topi = decoder_output.data.topk(TOP_K)
        possible_words = [self.get_word(v) for v in topi.view(-1)]
        return possible_words, decoder_hidden

    def simulate(self, max_length=50):
        sentence = input('Choose a sentence:')

        decoder_hidden = self.encode_sentence(sentence)
        prev_word = None           

        for _ in range(max_length):
            pred, decoder_hidden = self.predict_next_word(decoder_hidden, prev_word=prev_word)
            print(pred)
            prev_word = input('Choose your word or [stop]')
            if prev_word == '[stop]':
                break


if __name__ == "__main__":
    predictor = NextWordPredictor('models/encoder.pkl', 'models/decoder.pkl',
                                 'models/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin')
    predictor.simulate()

