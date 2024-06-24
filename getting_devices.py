import pyaudio

p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"Device {i}: {info['name']}, Input channels: {info['maxInputChannels']}, Output channels: {info['maxOutputChannels']}")

p.terminate()

# import numpy as np

# CHUNK = 1024  # Number of frames per buffer

# def callback(in_data, frame_count, time_info, status):
#     # Process audio data here (optional)
#     return (in_data, pyaudio.paContinue)

# p = pyaudio.PyAudio()

# # Open stream with Soundflower as input
# stream = p.open(format=pyaudio.paInt16,
#                 channels=2,  # Adjust based on Soundflower configuration
#                 rate=44100,  # Sample rate (adjust if necessary)
#                 input=True,
#                 input_device_index=4,  # Replace with Soundflower device index
#                 frames_per_buffer=CHUNK,
#                 stream_callback=callback)

# print("Capturing audio from Soundflower. Press Ctrl+C to stop.")

# # Start stream
# stream.start_stream()

# try:
#     while stream.is_active():
#         pass
# except KeyboardInterrupt:
#     print("Stopping...")

# # Stop stream
# stream.stop_stream()
# stream.close()

# # Terminate PyAudio
# p.terminate()
