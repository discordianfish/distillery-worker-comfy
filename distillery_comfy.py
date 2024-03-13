import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
from websocket import WebSocket # note: websocket-client (https://github.com/websocket-client/websocket-client)
import io
import requests
import time
import os
import subprocess
import tempfile
from typing import List
from distillery_aws import AWSConnector
import random

APP_NAME = os.getenv('APP_NAME') # Name of the application
API_COMMAND_LINE = os.getenv('API_COMMAND_LINE') # Command line to start the API server, e.g. "python3 ComfyUI/main.py"; warning: do not add parameter --port as it will be passed later
API_URL = os.getenv('API_URL')  # URL of the API server (warning: do not add the port number to the URL as it will be passed later)
INITIAL_PORT = int(os.getenv('INITIAL_PORT')) # Initial port to use when starting the API server; may be changed if the port is already in use
INSTANCE_IDENTIFIER = APP_NAME+'-'+str(uuid.uuid4()) # Unique identifier for this instance of the worker
MAX_COMFY_START_ATTEMPTS = 20  # Set this to the maximum number of attempts you want
MAX_SEED_INT=2147483647 # 2^31-1 to avoid overflow issues

TEST_PAYLOAD = json.load(open(os.getenv('TEST_PAYLOAD'))) # The TEST_PAYLOAD is a JSON object that contains a prompt that will be used to test if the API server is running
TEST_PAYLOAD["22"]["noise_seed"] = random.randint(0, MAX_SEED_INT) # Set a random noise seed for the test prompt

class ComfyConnector:
    _instance = None
    _process = None

    def __new__(cls, *args, **kwargs):
        try:
            if cls._instance is None:
                cls._instance = super(ComfyConnector, cls).__new__(cls)
            return cls._instance
        except Exception as e:
            raise

    def __init__(self):
        try:
            if not hasattr(self, 'initialized'):
                self.urlport = self.find_available_port()
                self.server_address = f"http://{API_URL}:{self.urlport}"
                self.client_id = INSTANCE_IDENTIFIER
                self.ws_address = f"ws://{API_URL}:{self.urlport}/ws?clientId={self.client_id}"
                self.ws = WebSocket()
                self.start_api()
                self.initialized = True
        except Exception as e:
            raise

    def find_available_port(self): # If the initial port is already in use, this method finds an available port to start the API server on
        try:
            port = INITIAL_PORT
            while True:
                try:
                    response = requests.get(f'http://{API_URL}:{port}')
                    if response.status_code != 200:
                        return port
                    else:
                        port += 1
                except requests.ConnectionError:
                    return port
        except Exception as e:
            raise
    
    def start_api(self): # This method is used to start the API server
        try:
            if not self.is_api_running(): # Block execution until the API server is running
                aws_connector = AWSConnector()
                api_command_line = API_COMMAND_LINE + f" --port {self.urlport}" # Add the port to the command line
                if self._process is None or self._process.poll() is not None: # Check if the process is not running or has terminated for some reason
                    self._process = subprocess.Popen(api_command_line.split())
                    aws_connector.print_log('N/A', INSTANCE_IDENTIFIER, f"ComfyUI startup began with PID: {self._process.pid} in port {self.urlport}", level='INFO')
                    attempts = 0
                    while not self.is_api_running(): # Block execution until the API server is running
                        if attempts >= MAX_COMFY_START_ATTEMPTS:
                            aws_connector.print_log('N/A', INSTANCE_IDENTIFIER, f"API startup procedure failed after {attempts} attempts.", level='ERROR')
                            raise RuntimeError(f"API startup procedure failed after {attempts} attempts.")
                        time.sleep(0.75)  # Wait for 1 second before checking again
                        attempts += 1 # Increment the number of attempts
                    aws_connector.print_log('N/A', INSTANCE_IDENTIFIER, f"ComfyUI startup successful after {attempts} attempts with PID: {self._process.pid} in port {self.urlport}", level='INFO')
                    time.sleep(0.25)  # Wait for 0.25 seconds before returning
        except Exception as e:
            raise

    def is_api_running(self): # This method is used to check if the API server is running
        test_payload = TEST_PAYLOAD
        try:
            response = requests.get(self.server_address)
            if response.status_code == 200: # Check if the API server tells us it's running by returning a 200 status code
                self.ws.connect(self.ws_address)
                test_image = self.generate_images(test_payload)
                if test_image:  # this ensures that the API server is actually running and not just the web server
                    return True
                return False
        except Exception as e:
            return False

    def kill_api(self): # This method is used to kill the API server
        try:
            if self._process is not None and self._process.poll() is None:
                aws_connector = AWSConnector()
                self._process.kill()
                self._process = None
                aws_connector.print_log('N/A', INSTANCE_IDENTIFIER, f"API process killed.", level='INFO')
                print("DISTILLERYPRINT: API process killed")
                self.cleanup()
        except Exception as e:
            raise

    def cleanup(self):
        # Close WebSocket connection with exception handling
        try:
            aws_connector = AWSConnector()
            if self.ws:
                try:
                    if self.ws.connected:
                        self.ws.close()
                except Exception as e:
                    aws_connector.print_log('N/A', INSTANCE_IDENTIFIER, f"API process killed.", level='INFO')
                finally:
                    self.ws = None
            # Reset other instance-specific attributes
            self.urlport = None
            self.server_address = None
            self.client_id = None
            # Reset the singleton instance
            ComfyConnector._instance = None
            print("DISTILLERYPRINT: ComfyConnector instance cleaned up")
        except Exception as e:
            raise

    def get_history(self, prompt_id): # This method is used to retrieve the history of a prompt from the API server
        try:
            with urllib.request.urlopen(f"{self.server_address}/history/{prompt_id}") as response:
                return json.loads(response.read())
        except Exception as e:
            raise

    def get_image(self, filename, subfolder, folder_type): # This method is used to retrieve an image from the API server
        try:
            data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
            url_values = urllib.parse.urlencode(data)
            with urllib.request.urlopen(f"{self.server_address}/view?{url_values}") as response:
                return response.read()
        except Exception as e:
            raise

    def queue_prompt(self, prompt): # This method is used to queue a prompt for execution
        try:
            p = {"prompt": prompt, "client_id": self.client_id}
            data = json.dumps(p).encode('utf-8')
            headers = {'Content-Type': 'application/json'}  # Set Content-Type header
            req = urllib.request.Request(f"{self.server_address}/prompt", data=data, headers=headers)
            return json.loads(urllib.request.urlopen(req).read())
        except Exception as e:
            raise

    def generate_images(self, payload): # This method is used to generate images from a prompt and is the main method of this class
        try:
            print(f"DISTILLERYPRINT: Generating images. Payload: {payload}")
            if not self.ws.connected: # Check if the WebSocket is connected to the API server and reconnect if necessary
                print("DISTILLERYPRINT: WebSocket is not connected. Reconnecting...")
                self.ws.connect(self.ws_address)
            prompt_id = self.queue_prompt(payload)['prompt_id']
            while True:
                out = self.ws.recv() # Wait for a message from the API server
                if isinstance(out, str): # Check if the message is a string
                    message = json.loads(out) # Parse the message as JSON
                    if message['type'] == 'executing': # Check if the message is an 'executing' message
                        data = message['data'] # Extract the data from the message
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            break
            address = self.find_output_node(payload) # Find the SaveImage node; workflow MUST contain only one SaveImage node
            history = self.get_history(prompt_id)[prompt_id]
            filenames = eval(f"history['outputs']{address}")['images']  # Extract all images
            images = []
            for img_info in filenames:
                filename = img_info['filename']
                subfolder = img_info['subfolder']
                folder_type = img_info['type']
                image_data = self.get_image(filename, subfolder, folder_type)
                image_file = io.BytesIO(image_data)
                image = Image.open(image_file)
                images.append(image)
            return images
        except Exception as e:
            raise

    def upload_image(self, filepath, subfolder=None, folder_type=None, overwrite=False):
        try: 
            url = f"{self.server_address}/upload/image"
            with open(filepath, 'rb') as file:
                files = {'image': file}
                data = {'overwrite': str(overwrite).lower()}
                if subfolder:
                    data['subfolder'] = subfolder
                if folder_type:
                    data['type'] = folder_type
                response = requests.post(url, files=files, data=data)
            return response.json()
        except Exception as e:
            raise

    @staticmethod
    def find_output_node(json_object): # This method is used to find the node containing the SaveImage class in a prompt
        try:
            for key, value in json_object.items():
                if isinstance(value, dict):
                    if value.get("class_type") == "SaveImage":
                        return f"['{key}']"  # Return the key containing the SaveImage class
                    result = ComfyConnector.find_output_node(value)
                    if result:
                        return result
            return None
        except Exception as e:
            raise
    
    @staticmethod
    def load_payload(path):
        try:
            with open(path, 'r') as file:
                return json.load(file)
        except Exception as e:
            raise

    def upload_from_s3_to_input(self, aws_connector, s3_keys: List[str]):
        try:
            file_objs = aws_connector.download_fileobj(s3_keys) # Download file objects from AWS S3
            for s3_key, file_obj in zip(s3_keys, file_objs): # Iterate through the downloaded file objects and corresponding S3 keys
                temp_file_path = os.path.join(tempfile.gettempdir(), os.path.basename(s3_key)) # Create a temporary file with the same name as the S3 key
                with open(temp_file_path, 'wb') as temp_file:
                    temp_file.write(file_obj.read())
                response = self.upload_image(filepath=temp_file_path, folder_type='input') # Upload the temporary file to the Comfy API in the 'input' folder
                os.unlink(temp_file_path) # Delete the temporary file
        except Exception as e:
            raise

