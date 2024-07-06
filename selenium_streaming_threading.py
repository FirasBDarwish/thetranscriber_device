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
from dotenv import dotenv_values
import google.generativeai as genai

import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from collections import deque

logging.basicConfig(level=20)
config = dotenv_values(".env")

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 3 # this indicates mono audio (single channel), stereo audio has more than one channel and is closed to surround sound
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS):
        def proxy_callback(in_data, frame_count, time_info, status):
            temp_stream = np.frombuffer(in_data, dtype=np.int16)
            temp_stream = np.reshape(temp_stream, (-1, self.CHANNELS))
            temp_stream = np.sum(temp_stream, axis=1) / self.CHANNELS # convert from multi-channel to mono
            in_data = temp_stream.astype(np.int16).tobytes()
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

    def vad_collector(self, padding_ms=700, ratio=0.75, frames=None):
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

def streaming(meetscode: str, model=None, sample_rate=16000, vad_aggressiveness=3, audio='Aggregate Device'):
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
    genai.configure(api_key=config["GOOGLE_API_KEY"])
    llm = genai.GenerativeModel('gemini-1.5-flash')

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

    # Flag to control the loop execution
    process_frames_flag = threading.Event()
    process_frames_flag.set()  # Initially set to continue processing
    # Initialize an Event object
    stop_event = threading.Event()    
    command_queue = queue.Queue() # Create a thread-safe queue for user commands
    context = ""
    context_lock = threading.Lock()

    # Function to get user input and control loop execution
    def get_user_input(command_queue, llm, meetscode):

        # Set up Google Meets connection with Meets
        opt = Options()
        opt.add_argument('--disable-blink-features=AutomationControlled')
        opt.add_argument('--start-maximized')
        opt.add_experimental_option("prefs", {
        
            "profile.default_content_setting_values.media_stream_mic": 0,
            "profile.default_content_setting_values.media_stream_camera": 0,
            "profile.default_content_setting_values.geolocation": 0,
            "profile.default_content_setting_values.notifications": 0
        })

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opt)
        driver.get("https://meet.google.com/" + meetscode)
        # driver.get("https://meet.google.com/")
        print("launched")

         # Wait up to 10 seconds for the notifications about mic and camera being off to become clickable
        button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[text()='Continue without microphone and camera']/ancestor::button"))
        )
        button.click()

        # Find the input field by its ID
        # input_field = driver.find_element(By.ID, 'c28')
        input_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "c22"))
        )
        # Clear any existing text in the input field
        input_field.clear()
        # Enter your name into the input field
        input_field.send_keys('TheTranscriber')
        # Send ENTER key to submit the form (if applicable)
        input_field.send_keys(Keys.ENTER)

        # Explicitly wait for the button to be clickable
        button = WebDriverWait(driver, 40).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Chat with everyone']"))
        )
        # Click the button
        button.click()

        message_text = "Hello, world! I am TheTranscriber, your personal.. transcriber! (didn't expect that, did you?). Feel free to ask me anything about the conversation so far. Just start your message with '@TheTranscriber' and ask!\n\nIf you want to pause (don't want me listening), type '@TheTranscriber p'.\nIf you want to resume the transcription, type '@TheTranscriber r'.\nIf you want to me to quit, type '@TheTranscriber q'."
        # Find the textarea by its ID (bfTqV in this case)
        textarea = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//textarea[@jsname='YPqjbf' and @class='qdOxv-fmcmS-wGMbrd xYOaDe' and @aria-label='Send a message' and @placeholder='Send a message']"))
        )
        # Clear any existing text in the textarea (if needed)
        textarea.clear()
        # Enter text into the textarea
        textarea.send_keys(message_text)

        # Find the button to send the message by its XPath
        send_button = driver.find_element(By.XPATH, "//button[@aria-label='Send a message']")
        # Click the send button
        send_button.click()
        messages_received = deque()
        messages_received_index = 1
        nonlocal context
        quitting=False
        while True:
            messages = driver.find_elements(By.XPATH, "//div[@jscontroller='RrV5Ic']")
            if(len(messages)-messages_received_index > 0):
                for iter in range(len(messages)-messages_received_index):
                    if(messages[messages_received_index].text[0:15] == "@TheTranscriber"):
                        if messages[messages_received_index].text == "@TheTranscriber p" or messages[messages_received_index].text == "@TheTranscriber r":
                            command_queue.put(messages[messages_received_index].text[-1])
                        elif messages[messages_received_index].text == "@TheTranscriber q":
                            # print("before")
                            command = messages[messages_received_index].text[-1]
                            driver.quit()
                            # print("after")
                            command_queue.put(command)
                            quitting=True
                            break
                        else:
                            messages_received.append(messages[messages_received_index].text[15:])
                    messages_received_index+=1
                if quitting:
                    break
            if(len(messages_received) > 0):
                user_query = messages_received.popleft()
                with context_lock:
                    print("Context:", context)
                    prompt = ("You are given a transcript of a conversation. Afterwards, you will be asked a question about the conversation, please respond based on the information in the transcript.\n Transcript: " + context + "\n\n" + user_query)
                response = llm.generate_content(prompt)
                reply_text = user_query + "\n\nTheTranscriber: " + response.text
                textarea.clear()
                # Enter text into the textarea
                textarea.send_keys(reply_text)
                send_button.click()
                print(response.text)

    # Main function to process frames
    def process_frames(frames, stream_context, stop_event):
        nonlocal context
        for frame in frames:
            if stop_event.is_set():
                break
            if process_frames_flag.is_set():  # Wait if flag is cleared (paused)
                if frame is not None:
                    # if spinner: spinner.start()
                    logging.debug("streaming frame")
                    stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
                else:
                    # if spinner: spinner.stop()
                    logging.debug("end utterence")
                    text = stream_context.finishStream()
                    # print("Recognized: %s" % text)
                    with context_lock:
                        context+=("\n" + text)
                    # print("context:", context)
                    stream_context = model.createStream()
            else:
                continue

        
    
    # Start the user input thread
    input_thread = threading.Thread(target=get_user_input, args=(command_queue, llm, meetscode))
    input_thread.start()

    # Start processing frames
    process_frames_thread = threading.Thread(target=process_frames, args=(frames, stream_context, stop_event))
    process_frames_thread.start()

    # Main thread to handle user commands and control loop execution
    while True:
        command = command_queue.get()
        if command == 'p':
            process_frames_flag.clear()  # Pause the processing loop
            print("Processing paused.")
        elif command == 'r':
            process_frames_flag.set()  # Resume the processing loop
            print("Processing resumed.")
        elif command == 'q':
            process_frames_flag.set()  # Resume the processing loop
            stop_event.set()  # Pause the processing loop
            print("Ending...")
            break  # Exit the program

    # Wait for threads to complete
    process_frames_thread.join()
    input_thread.join()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="TheTranscriber: your personal listening and responding GoogleMeets bot")
    parser.add_argument('--meetscode', help='Specify the meetscode value (format: aaa-aaa-aaa)')
    
    args = parser.parse_args()
    if args.meetscode:
        streaming(args.meetscode, 'deepspeech-0.9.3-models.pbmm', 44100)
    else:
        print("Error: No GoogleMeets code provided. Ending..")