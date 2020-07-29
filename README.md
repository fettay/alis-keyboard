# ALIS Keyboard

Smart Keyboard made in a Hackathon for people with Locked In Syndrome.
To setup it requires to download the embedding [frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin](http://embeddings.net/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin) and put the file in the models directory.
Then you need to create a config.py file with:
```python
SPEECH_KEY = '' # A key from azure cognitive services
SPEECH_REGION = '' # The region of azure cognitive services
```
