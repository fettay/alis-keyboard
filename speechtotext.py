import azure.cognitiveservices.speech as speechsdk
from config import SPEECH_KEY, SPEECH_REGION


LANGUAGE = 'fr-FR'

class SpeechToText:

    # Creates an instance of a speech config with specified subscription key and service region.
    # Replace with your own subscription key and service region (e.g., "westus").
    def __init__(self):
        speech_key, service_region = SPEECH_KEY, SPEECH_REGION
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region,
                                               speech_recognition_language=LANGUAGE)
        speech_config.speech_synthesis_language = LANGUAGE
        self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    def recognize(self):
        result = self.speech_recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        else:
            return ""

    def talk(self, sentence):
        self.speech_synthesizer.speak_text(sentence)


if __name__ == "__main__":
    sp = SpeechToText()

    # print('Say something')
    # print(sp.recognize())

    print(sp.talk('salut, tu vas bien ?'))