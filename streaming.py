import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import deepspeech
import numpy as np
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal
import sys

logging.basicConfig(level=20)

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1 # this indicates mono audio (single channel), stereo audio has more than one channel and is closed to surround sound
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS):
        def proxy_callback(in_data, frame_count, time_info, status):
            callback(in_data)
            return (None, pyaudio.paContinue)

        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data) # callback is usually passed in as none (so pretty much in our work here self.buffer_queue holds all of our audio)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND)) # in the case taht self.RATE_PROCESS (default sampling rate) is different from self.input_rate
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate, # which, again, could be different from self.RATE_PROCESS (default sampling rate)
            'input': True,
            'output': False, # output is if I want pyaudio to play audio to me (not pick up my system audio, need to figure this out)
            'frames_per_buffer': self.block_size_input, # buffer is practically a block
            'stream_callback': proxy_callback,
        }

        # if not default device (if using gpu, I suppose)
        if self.device:
            kwargs['input_device_index'] = self.device

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    def write_wav(self, filename, data):
        logging.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()

class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None):
        super().__init__(device=device, input_rate=input_rate)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None: frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames) # appemdomg new elements when the deque is full leads to left-most elements being popped (newest element pops out oldest element)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()

def streaming(model=None, sample_rate=16000, vad_aggressiveness=3, audio='Aggregate Device'):
    """
    Function that streams audio from audio device (depending on how audio is set), recognizes
    human voice, and creates a transcription.
    
    Args:
        model: Deepspeech model that you would like to use to process input.
        sample_rate: sample_rate (Aggregate Device on Mac is 44100Hz). Default value is 16000Hz
        vad_aggressiveness: How aggressive the baseline for detecting human voice activity should be.
        audio: Name of audio device you would like your pyAudio to pick up. You can run the 
        'getting_devices.py' script as python getting_device.py to get a list of devices
        recognized by PyAudio and see which one you would like to use.

    In order to use Aggregate Device on macOS:
        1. Install Soundflower and ensure that you have a Soundflower (2 ch) output device
        in your settings.

        2. Go to the application "Audio MIDI Setup" and add an 'Aggregate Device' that combines your
        regular output as well as your device's microphone

        Done! This will now allow PyAudio to recognize a device by the name 'Aggregate Device'
        that will allow it to process both audio from your microphone but also audio that is
        being outputted by your device (any video you are watching, zoom call you are in, etc.)
    """
    p = pyaudio.PyAudio()

    device = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['name'] == audio:
            device = i
        # print(f"Device {i}: {info['name']}, Input channels: {info['maxInputChannels']}, Output channels: {info['maxOutputChannels']}")

    if device is None:
        print("device not recognized, unrecognized behavior..", file=sys.stderr)

    p.terminate()

    # Load DeepSpeech model
    if os.path.isdir(model):
        model_dir = model
        model = os.path.join(model_dir, 'output_graph.pb')

    model = deepspeech.Model(model)

    # Start audio with VAD
    vad_audio = VADAudio(aggressiveness=vad_aggressiveness,
                         device=device,
                         input_rate=sample_rate)

    
    # print("Listening (ctrl-C to exit)...")
    frames = vad_audio.vad_collector()

    # Stream from microphone to DeepSpeech using VAD
    spinner = None
    spinner = Halo(spinner='line')
    stream_context = model.createStream()
    wav_data = bytearray()
    for frame in frames:
        if frame is not None:
            if spinner: spinner.start()
            logging.debug("streaming frame")
            stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
        else:
            if spinner: spinner.stop()
            logging.debug("end utterence")
            text = stream_context.finishStream()
            print("Recognized: %s" % text)
            stream_context = model.createStream()

if __name__ == '__main__':
    streaming('deepspeech-0.9.3-models.pbmm', 44100)