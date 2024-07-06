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
import argparse

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
import regex as re

config = dotenv_values(".env")

def connect_meet(meetscode):
    
    """
    Function to connect to Google Meets.

    Args:
        code: string in form xxx-xxx-xxx representing the Google Meets meeting code that you wish to connect to
    Return:
        driver: driver to continue communicating with Meets
    """
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
    button_notification = WebDriverWait(driver, 40).until(
        EC.element_to_be_clickable((By.XPATH, "//span[text()='Continue without microphone and camera']/ancestor::button"))
    )
    button_notification.click()

    # Find the input field by its ID
    # input_field = driver.find_element(By.ID, 'c22')
    input_field = WebDriverWait(driver, 40).until(
        EC.presence_of_element_located((By.ID, "c22"))
    )
    # Clear any existing text in the input field
    input_field.clear()
    # Enter your name into the input field
    input_field.send_keys('TheTranscriber')
    # Send ENTER key to submit the form (if applicable)
    input_field.send_keys(Keys.ENTER)

    button_captions = WebDriverWait(driver, 40).until(
        EC.element_to_be_clickable((By.XPATH, '//button[@aria-label="Turn on captions"]'))
    )
    button_captions.click()

    # Explicitly wait for the button to be clickable
    button = WebDriverWait(driver, 40).until(
        EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Chat with everyone']"))
    )
    # Click the button
    button.click()

    message_text = "Hello, world! I am TheTranscriber, your personal.. transcriber! (didn't expect that, did you?). Feel free to ask me anything about the conversation so far. Just start your message with '@TheTranscriber' and ask!\n\nIf you want to pause (don't want me listening), type '@TheTranscriber p'.\nIf you want to resume the transcription, type '@TheTranscriber r'.\nIf you want to me to quit, type '@TheTranscriber q'."
    # Find the textarea by its ID (bfTqV in this case)
    textarea = WebDriverWait(driver, 40).until(
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

    # Connected
    return driver, textarea, send_button

def _fusion(s1, s2):
    """
    Helper function to fuse strings
    """
    for m in re.finditer(s2[0], s1):
        if s2.startswith(s1[m.start():]):
            return True, s1[:m.start()] + s2
    # no overlap found
    return False, ""

def transcriber(driver, textarea, send_button, llm):
    messages_received = deque()
    command_queue = queue.Queue()
    context = ""
    context_dic = {}
    listening = True

    messages_received_index = 1
    quitting=False

    # Locate the div element that contains the content to monitor
    main_div = driver.find_element(By.CLASS_NAME, "iOzk7")
    print("Found")
    # Track initial inner HTML for comparison
    initial_inner_html = main_div.get_attribute("innerHTML")

    while True:
        # Check the current inner HTML of the div element
        current_inner_html = main_div.get_attribute("innerHTML")
        if listening and (current_inner_html != initial_inner_html):
            # print("Changes detected:")
            # Find all div elements with class 'TBMuR bj4p3b' within the main div
            internal_divs = main_div.find_elements(By.CSS_SELECTOR, ".TBMuR.bj4p3b")
            for div in internal_divs:
                # Find divs with class 'zs7s8d jxFHg' within each div with class 'TBMuR bj4p3b'
                username = ""
                try:
                    username_div = div.find_element(By.CSS_SELECTOR, ".zs7s8d.jxFHg")
                    username = username_div.text.strip()
                except:
                    pass
                try:
                    context_dic[username]
                except KeyError:
                    context_dic[username] = []

                # Find spans under div with jsname 'tgaKEf' and class 'iTTPOb VbkSUe'
                time.sleep(1)
                spans=[]
                try:
                    spans = div.find_elements(By.CSS_SELECTOR, "div[jsname='tgaKEf'].iTTPOb.VbkSUe span")
                except:
                    pass
                to_add_context = ""
                for span in spans:
                    try:
                        span = span.text.strip()
                        if span not in context_dic[username]:
                            context_dic[username].append(span)
                            to_add_context += span
                    except:
                        pass
                if(len(to_add_context) > 0):
                    last_line = context.strip().split("\n")[-1]
                    if(len(last_line)>0 and last_line[-1]=='.'):
                        last_line = last_line[:-1]
                    to_add_context = to_add_context[:1].lower() + to_add_context[1:]                    
                    state, stro = _fusion(last_line, to_add_context)
                    if state:
                        lst = context.strip().split("\n")
                        lst[-1] = stro
                        # print("stro:", stro)
                        context = "\n".join(lst)
                    else:
                            context+=("\n" + username + ": " + to_add_context)

        messages = driver.find_elements(By.XPATH, "//div[@jscontroller='RrV5Ic']")
        if(len(messages)-messages_received_index > 0):
            for iter in range(len(messages)-messages_received_index):
                if(messages[messages_received_index].text[0:15] == "@TheTranscriber"):
                    if listening and messages[messages_received_index].text == "@TheTranscriber p":
                        listening=False
                    elif not listening and messages[messages_received_index].text == "@TheTranscriber r":
                        listening=True
                    elif messages[messages_received_index].text == "@TheTranscriber q":
                        driver.quit()
                        quitting=True
                        break
                    else:
                        if(listening):
                            messages_received.append(messages[messages_received_index].text[15:])
                messages_received_index+=1
            if quitting:
                break
        if(listening and len(messages_received) > 0):
            user_query = messages_received.popleft()
            # print("Context:", context)
            prompt = ("You are given a transcript of a conversation. Afterwards, you will be asked a question about the conversation, please respond based on the information in the transcript.\n Transcript: " + context + "\n\n" + user_query)
            response = llm.generate_content(prompt)
            reply_text = user_query + "\n\nTheTranscriber: " + response.text
            textarea.clear()
            # Enter text into the textarea
            textarea.send_keys(reply_text)
            send_button.click()
            # print(response.text)
    
    print("Ending..")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="TheTranscriber: your personal listening and responding GoogleMeets bot")
    parser.add_argument('--meetscode', help='Specify the meetscode value (format: aaa-aaa-aaa)')
    
    args = parser.parse_args()
    if args.meetscode:
        driver, textarea, send_button = connect_meet(args.meetscode)
        # Load DeepSpeech model

        model = 'deepspeech-0.9.3-models.pbmm'
        if os.path.isdir(model):
            model_dir = model
            model = os.path.join(model_dir, 'output_graph.pb')

        model = deepspeech.Model(model)
        genai.configure(api_key=config["GOOGLE_API_KEY"])
        llm = genai.GenerativeModel('gemini-1.5-flash')
        transcriber(driver, textarea, send_button, llm)
    else:
        print("Error: No GoogleMeets code provided. Ending..")

    