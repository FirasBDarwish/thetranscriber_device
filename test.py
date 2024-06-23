from deepspeech import Model
import numpy as np
import speech_recognition as sr


sample_rate = 16000
beam_width = 500
lm_alpha = 0.75
lm_beta = 1.85
n_features = 26
n_context = 9

model_name = "deepspeech-0.9.3-models.pbmm"
alphabet = "alphabet.txt"
langauage_model = "lm.binary"
trie = "trie"
audio_file = "demo.wav"


if __name__ == '__main__':
    # ds = Model(model_name, n_features, n_context, alphabet, beam_width)
    ds = Model(model_name)
    # ds.enableDecoderWithLM(alphabet, langauage_model, trie, lm_alpha, lm_beta)

    r = sr.Recognizer()
    with sr.Microphone(sample_rate=sample_rate) as source:
        print("Say Something")
        audio = r.listen(source)
        fs = audio.sample_rate
        audio = np.frombuffer(audio.frame_data, np.int16)



    #fin = wave.open(audio_file, 'rb')
    #fs = fin.getframerate()
    #print("Framerate: ", fs)

    #audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    #audio_length = fin.getnframes() * (1/sample_rate)
    #fin.close()

    print("Infering {} file".format(audio_file))

    print(ds.stt(audio))