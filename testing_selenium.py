# from selenium import webdriver
# from webdriver_manager.chrome import ChromeDriverManager

# import time

# # Google Meet URL
# meet_url = 'https://meet.google.com/'

# # Replace with your meeting code
# meeting_code = 'fxs-asvs-vhs'

# try:
#     # Initialize Chrome WebDriver
#     # options = webdriver.ChromeOptions()
#     # cService = ChromeDriverManager().install()
#     driver = webdriver.Chrome()


#     # Open Google Meet
#     driver.get(meet_url)

#     # Wait for some time to load the page
#     time.sleep(5)

#     # Find and click on the 'Join or start a meeting' button
#     join_button = driver.find_element_by_xpath("//a[@href='/link/meetingcode']")
#     join_button.click()

#     # Find the input field to enter the meeting code
#     code_input = driver.find_element_by_xpath("//input[@id='i3']")
#     code_input.send_keys(meeting_code)

#     # Click on the 'Continue' button
#     continue_button = driver.find_element_by_xpath("//span[text()='Continue']")
#     continue_button.click()

#     # Wait for some time to join the meeting
#     time.sleep(5)

#     # Find and click on the 'Join now' button
#     join_now_button = driver.find_element_by_xpath("//span[text()='Join now']")
#     join_now_button.click()

#     # Wait for the meeting to join
#     time.sleep(10)  # Adjust the waiting time as needed

# except Exception as e:
#     print(f"Error: {e}")

# finally:
#     # Close the browser
#     driver.quit()

import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

code = "ywn-hxkp-scv"
# chrome_options = Options()
# chrome_options.add_argument("--use-fake-ui-for-media-stream")  # Use fake UI to allow permissions automatically
opt = Options()
opt.add_argument('--disable-blink-features=AutomationControlled')
opt.add_argument('--start-maximized')
opt.add_experimental_option("prefs", {
 
    "profile.default_content_setting_values.media_stream_mic": 1,
    "profile.default_content_setting_values.media_stream_camera": 1,
    "profile.default_content_setting_values.geolocation": 0,
    "profile.default_content_setting_values.notifications": 1
})

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opt)

try:
    driver.get("https://meet.google.com/" + code)
    # driver.get("https://meet.google.com/")
    print("launched")

    # Find the input field by its ID
    # input_field = driver.find_element(By.ID, 'c28')
    input_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "c28"))
    )
    # Clear any existing text in the input field
    input_field.clear()
    # Enter your name into the input field
    input_field.send_keys('TheTranscriber')
    # Send ENTER key to submit the form (if applicable)
    input_field.send_keys(Keys.ENTER)

    # Explicitly wait for the button to be clickable
    button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Chat with everyone']"))
    )
    # Click the button
    button.click()

    message_text = "Hello, world! I am TheTranscriber, your personal.. transcriber! (didn't expect that, did you?). Feel free to ask me anything about the conversation so far. Just type '@TheTranscriber ..."
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

    messages_text = []
    while(True): # accidentally picking up its own messages!, use @TheTranscriber, also use a queue to play with
        messages = driver.find_elements(By.XPATH, "//div[@jscontroller='RrV5Ic']")
        messages_text = [message.text for message in messages]
        print(messages_text)
        time.sleep(5)

    while(True):
        pass # temporary solution to keep it indefinitely running

finally:
    # Close the browser
    driver.quit()
