import time
import runpod
import uuid
from distillery_aws import AWSConnector
from distillery_comfy import ComfyConnector
import os
import io
from urllib.parse import urlparse
from PIL import PngImagePlugin
import json
from concurrent.futures import ThreadPoolExecutor
import copy
import sys
import subprocess
import traceback
import logging
from distillery_train import do_training
import shutil
import random
import string

MAX_WORKER_ATTEMPTS = 2 # Maximum number of times the worker will attempt to run before giving up
START_TIME = time.time() # Time at which the worker was initialized
APP_NAME = os.getenv('APP_NAME') # Name of the application
INSTANCE_IDENTIFIER = APP_NAME+'-'+str(uuid.uuid4()) # Unique identifier for this instance of the worker
NETWORK_STORAGE = os.getenv("NETWORK_STORAGE") # Path to network storage mount
MODELS_FOLDER = os.getenv("MODELS_FOLDER") # Path to models folder in ComfyUI
INFERENCE_OUTPUT_FOLDER = os.getenv("INFERENCE_OUTPUT_FOLDER") # Path to output folder in ComfyUI
INFERENCE_INPUT_FOLDER = os.getenv("INFERENCE_INPUT_FOLDER") # Path to input folder in ComfyUI
CUSTOM_NODES_FOLDER = os.getenv("CUSTOM_NODES_FOLDER") # Path to custom nodes folder in ComfyUI
WORKER_TIMEOUT_FOR_INFERENCE = int(os.getenv("WORKER_TIMEOUT_FOR_INFERENCE")) # Timeout for the worker in seconds for inference
WORKER_TIMEOUT_FOR_TRAINING = int(os.getenv("WORKER_TIMEOUT_FOR_TRAINING")) # Timeout for the worker in seconds for training
MINIMUM_GB_FREE_DISK_SPACE = int(os.getenv("MINIMUM_GB_FREE_DISK_SPACE")) # Minimum GB of free disk space required for the worker to run

def send_runpod_errorlog(preamble_text, request_id):
    def clean_repr(obj):
        return repr(obj).replace('\\n', ' ').replace('\\t', ' ')
    etype, value, tb = sys.exc_info()  # Get the exception information
    if etype is None:  # Ensure there is an exception to process
        return    
    single_line_traceback = ' -> '.join(traceback.format_exception(etype, value, tb)).replace('\n', ' ').replace('\t', ' ').strip() # Formatting the traceback into a single line, replacing newlines and tabs
    local_vars = {}  # Get local variables from the caller's frame
    if tb is not None:
        frame = tb.tb_frame
        local_vars = frame.f_locals
    formatted_locals = ', '.join(f"{k}: {clean_repr(v)}" for k, v in local_vars.items() if k != '__builtins__') # Format local variables, replacing newlines and tabs in their string representations
    complete_traceback = f'{preamble_text}: ' + single_line_traceback + ' | Local Variables: ' + formatted_locals
    print(complete_traceback)
    logging.error(complete_traceback) # Log the formatted single line traceback
    aws_connector = AWSConnector()
    aws_connector.print_log(request_id, INSTANCE_IDENTIFIER, complete_traceback, level='ERROR')
    return complete_traceback

def confirm_disk_space():
    # declare helper function to get free space
    def get_free_space_gb(folder): # Return folder/drive free space (in gigabytes)
        total, used, free = shutil.disk_usage(folder)
        return free // (2**30)
    # declare helper function to delete files from a folder
    def delete_contents(folder): # Delete the contents of the specified folder
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                raise
    if get_free_space_gb('/') < MINIMUM_GB_FREE_DISK_SPACE:  # Checking the root directory for overall disk space
        delete_contents(INFERENCE_OUTPUT_FOLDER)
        delete_contents(INFERENCE_INPUT_FOLDER)
        delete_contents(f"{MODELS_FOLDER}/loras/")

def fetch_images(payload):
    try:
        aws_connector = AWSConnector() 
        comfy_connector = ComfyConnector()
        image_files = []
        comfy_api = payload['comfy_api']
        request_id = payload['request_id']
        template_inputs = payload['template_inputs']
        print(f"DISTILLERYPRINT: Request ID {request_id} being processed with template inputs {template_inputs} and workflow {payload['payload_template_key']}.")
        images = comfy_connector.generate_images(comfy_api)
        for image in images: 
            # Create a unique filename
            filename = f'distillery_{str(uuid.uuid4())}.png'

            # Add the metadata
            image_metadata_dict = copy.deepcopy(payload)
            #del image_metadata_dict['comfy_api'] # Remove the Comfy API from the metadata to keep the size small
            image_metadata = json.dumps(image_metadata_dict)
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text('prompt', image_metadata)

            # Save the image to an in-memory file object
            image_file = io.BytesIO()
            image.save(image_file, format='PNG', pnginfo=pnginfo)
            image_file.seek(0)

            # Upload the in-memory file to S3
            aws_connector.upload_fileobj([(image_file, filename)]) # Upload the in-memory file to S3
            image_files.append(filename)

        return image_files # Return the list of keys of the images in S3
    except Exception as e:
        raise

class InputPreprocessor:
    @staticmethod
    def update_paths(json_obj, paths, input_value):
        try:
            aws_connector = AWSConnector()
            updated_json_obj = copy.deepcopy(json_obj)  # Create a deep copy of the original JSON object
            for path in paths:
                target = updated_json_obj
                for key in path[:-1]:  # Traverse all but the last key in the path
                    target = target.get(key, {})  # Use get to avoid KeyError
                if path[-1] in target:  # Check if the key exists
                    target[path[-1]] = input_value  # Update the value at the last key in the path
            return updated_json_obj
        except Exception as e:
            raise

    @staticmethod
    def tally_models_to_fetch(template_inputs):
        try:
            # let's build a list containing all models we need to fetch. 
            # For that, we first need to iterate over all template_inputs and check if the key is a model key.
            # For every model key, we will build a list of dictionaries containing the model name and model type, and having the comfy_input as key.
            models_to_fetch = []
            for key, value in template_inputs.items():
                if key in ['MODEL_CHECKPOINT_FILENAME', 'KSAMPLER_SEC_MODEL_FILENAME']:
                    comfy_input = key
                    model_filename = template_inputs[key]
                    model_type = "sd_model"
                    model_to_add = {comfy_input: {'model_filename': model_filename, 'model_type': model_type}}
                    models_to_fetch.append(model_to_add)
                elif key in ['LCM_LORA_FILENAME','LORA_1_FILENAME','LORA_2_FILENAME','LORA_3_FILENAME','LORA_4_FILENAME','LORA_5_FILENAME','LORA_6_FILENAME','LORA_7_FILENAME','LORA_8_FILENAME','LORA_9_FILENAME']:
                    comfy_input = key
                    model_filename = template_inputs[key]
                    model_type = "lora_model"
                    model_to_add = {comfy_input: {'model_filename': model_filename, 'model_type': model_type}}
                    models_to_fetch.append(model_to_add)
                elif key in ['CONTROLNET_MODEL_FILENAME','CONTROLNET_INPAINTING_MODEL_FILENAME']:
                    comfy_input = key
                    model_filename = template_inputs[key]
                    model_type = "controlnet_model"
                    model_to_add = {comfy_input: {'model_filename': model_filename, 'model_type': model_type}}
                    models_to_fetch.append(model_to_add)
                elif key in ['IPADAPTER_MODEL_FILENAME']:
                    comfy_input = key
                    model_filename = template_inputs[key]
                    model_type = "ipadapter_model"
                    model_to_add = {comfy_input: {'model_filename': model_filename, 'model_type': model_type}}
                    models_to_fetch.append(model_to_add)
            # Now, let's check if there are any errors in the list of models to fetch. If there are, we will print the error and delete the model from the list.
            if models_to_fetch:
                models_to_fetch_original = copy.deepcopy(models_to_fetch)
                for model in models_to_fetch_original:
                    for key, value in model.items():
                        if 'model_filename' not in value:
                            print(f"DISTILLERYPRINT: Error: {key} model_filename is not in value. Deleting model from list.")
                            models_to_fetch.remove(model)
                        else:
                            if value['model_filename'] is None:
                                print(f"Error: {key} model_filename is None. Deleting model from list.")
                                models_to_fetch.remove(model)
                        if 'model_type' not in value:
                            print(f"DISTILLERYPRINT: Error: {key} model_type is not in value. Deleting model from list.")
                            models_to_fetch.remove(model)
                        else:
                            if value['model_type'] is None:
                                print(f"DISTILLERYPRINT: Error: {key} model_type is None. Deleting model from list.")
                                models_to_fetch.remove(model)
            print(f"DISTILLERYPRINT: Models to fetch: {models_to_fetch}. Original list: {models_to_fetch_original}")
            # code here
            return models_to_fetch
        except Exception as e:
            raise

    @staticmethod
    def get_models_from_storage(models_list, request_id, save_to_network_storage = True):
        try:
            aws_connector = AWSConnector()
            start_time = time.time()
            copied_models = []
            copied_models_times = []
            total_models_processed = 0
            if models_list:
                for item in models_list:  # will iterate across all items in the list
                    for model in item.values():  # will iterate across all models in the list
                        copy_start_time = time.time()
                        if model["model_type"] == "sd_model":
                            model_type_path ="checkpoints"
                            model_path = f"{MODELS_FOLDER}/checkpoints/"
                        elif model["model_type"] == "lora_model":
                            model_type_path ="loras"
                            model_path = f"{MODELS_FOLDER}/loras/"
                        elif model["model_type"] == "controlnet_model":
                            model_type_path ="controlnet"
                            model_path = f"{MODELS_FOLDER}/controlnet/"
                        elif model["model_type"] == "ipadapter_model":
                            model_type_path ="ipadapter"
                            model_path = f"{CUSTOM_NODES_FOLDER}/ComfyUI_IPAdapter_plus/models/"
                        if model["model_filename"] not in os.listdir(model_path):
                            if model["model_filename"] in os.listdir(f"{NETWORK_STORAGE}/{model_type_path}"):
                                try:
                                    subprocess.run(["cp", f"{NETWORK_STORAGE}/{model_type_path}/{model['model_filename']}", f"{model_path}"], check=True)
                                except subprocess.CalledProcessError as e:
                                    subprocess_error_message = f"ERROR: Failed to copy model['model_filename'] '{model['model_filename']}' from network storage"
                                    complete_errorlog = send_runpod_errorlog(subprocess_error_message, request_id)
                                    raise subprocess_error_message
                            else:
                                try:
                                    aws_connector.download_files([(model["model_filename"], f"{model_path}/{model['model_filename']}")]) # Download the model from S3
                                    if save_to_network_storage: # Save the model to the Runpod database, once downloaded
                                        try:
                                            rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10)) # Generate a random string to append to the filename to avoid overwriting and allow concurrent workers to run
                                            file_name = model['model_filename']
                                            temporary_file_name = f"{rand}_{file_name}"
                                            subprocess.run(["cp", f"{model_path}/{model['model_filename']}", f"{NETWORK_STORAGE}/{model_type_path}/{temporary_file_name}"], check=True)
                                            os.rename(f"{NETWORK_STORAGE}/{model_type_path}/{temporary_file_name}", f"{NETWORK_STORAGE}/{model_type_path}/{file_name}")
                                            print()
                                        except Exception as e:
                                            complete_errorlog = send_runpod_errorlog(f"DISTILLERYPRINT: Warning - Failed to copy model to network storage: '{model['model_filename']}' from S3", request_id)
                                            pass
                                except Exception as e:
                                    complete_errorlog = send_runpod_errorlog(f"DISTILLERYPRINT: Error downloading model '{model['model_filename']}' from S3", request_id)
                                    raise
                            copy_end_time = time.time()
                            copied_models.append(model['model_filename'])
                            copied_models_times.append(copy_end_time - copy_start_time)
                        total_models_processed += 1
            end_time = time.time()
            total_time_consumed = end_time - start_time
            aws_connector.print_log(request_id, INSTANCE_IDENTIFIER, f"DISTILLERYPRINT: #{request_id} - {len(copied_models)} models copied from network storage in {total_time_consumed} seconds. Models copied: {copied_models}", level='INFO')
        except Exception as e:
            complete_errorlog = send_runpod_errorlog("Error in get_models_from_storage", request_id)
            raise complete_errorlog

def flatten_list(nested_list):
    try:
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(flatten_list(item))
            else:
                flat_list.append(item)
        return flat_list
    except Exception as e:
        raise

def worker_routine(event):
    payload_template_key = None
    comfy_connector = None
    template_inputs = None
    try:
        aws_connector = AWSConnector()
        payload = copy.deepcopy(event['input'])
        request_id = payload['request_id']
    except Exception as e:
        raise
    if 'request_type' in payload:
        if payload['request_type'] == 'distill':
            lora_name = payload['lora_name']
            original_image_file_name = payload['distill_image_filename']
            force_category = None
            if 'category' in payload['parsed_output']:
                force_category = payload['parsed_output']['category'] if payload['parsed_output']['category'] != 'autodetect' else None
            return do_training(lora_name, original_image_file_name, force_category=force_category)
    attempt_number = 1
    try:
        while attempt_number <= MAX_WORKER_ATTEMPTS:
            try:
                comfy_connector = ComfyConnector()
                if not 'input' in event:
                    aws_connector.print_log('N/A', INSTANCE_IDENTIFIER, f"Worker was passed a None payload from event.", level='ERROR')        
                    return None
                template_inputs = payload['template_inputs']
                images_per_batch = payload['images_per_batch']
                comfy_api = payload['comfy_api']
                noise_seed_template_paths = payload['noise_seed_template_paths']
                payload_template_key = payload['payload_template_key']
                # img2img
                if 'IMG2IMG_IMAGE_FILENAME' in template_inputs:
                    comfy_connector.upload_from_s3_to_input(aws_connector, [template_inputs['IMG2IMG_IMAGE_FILENAME']])
                # inpaint
                if 'INPAINT_IMAGE_FILENAME' in template_inputs:
                    comfy_connector.upload_from_s3_to_input(aws_connector, [template_inputs['INPAINT_IMAGE_FILENAME']])
                if 'INPAINT_MASK_IMAGE_FILENAME' in template_inputs:
                    comfy_connector.upload_from_s3_to_input(aws_connector, [template_inputs['INPAINT_MASK_IMAGE_FILENAME']])
                # controlnet
                if 'CONTROLNET_IMAGE_FILENAME' in template_inputs:
                    comfy_connector.upload_from_s3_to_input(aws_connector, [template_inputs['CONTROLNET_IMAGE_FILENAME']])
                # zoomout
                if 'ZOOM_OUT_IMAGE_FILENAME' in template_inputs:
                    comfy_connector.upload_from_s3_to_input(aws_connector, [template_inputs['ZOOM_OUT_IMAGE_FILENAME']])
                # IPAdapter
                if 'IPADAPTER_1_IMAGE_FILENAME' in template_inputs:
                    comfy_connector.upload_from_s3_to_input(aws_connector, [template_inputs['IPADAPTER_1_IMAGE_FILENAME']]) 
                if 'IPADAPTER_2_IMAGE_FILENAME' in template_inputs:
                    comfy_connector.upload_from_s3_to_input(aws_connector, [template_inputs['IPADAPTER_2_IMAGE_FILENAME']]) 
                if 'IPADAPTER_3_IMAGE_FILENAME' in template_inputs:
                    comfy_connector.upload_from_s3_to_input(aws_connector, [template_inputs['IPADAPTER_3_IMAGE_FILENAME']]) 
                if 'IPADAPTER_4_IMAGE_FILENAME' in template_inputs:
                    comfy_connector.upload_from_s3_to_input(aws_connector, [template_inputs['IPADAPTER_4_IMAGE_FILENAME']]) 
                models_to_fetch = InputPreprocessor.tally_models_to_fetch(template_inputs)
                if models_to_fetch:
                    InputPreprocessor.get_models_from_storage(models_to_fetch, request_id) # Copy models from network storage to ComfyUI
                files = []
                for i in range(images_per_batch):
                    file = fetch_images(payload)
                    files.append(file)
                    if isinstance(template_inputs['NOISE_SEED'], str):
                        template_inputs['NOISE_SEED']=str(int(template_inputs['NOISE_SEED'])+1)
                    else:
                        template_inputs['NOISE_SEED'] += 1
                    comfy_api = InputPreprocessor.update_paths(comfy_api, noise_seed_template_paths, template_inputs['NOISE_SEED'])
                    payload['comfy_api'] = comfy_api
                    print(f"DISTILLERYPRINT: Image {i+1} - New Seed: {template_inputs['NOISE_SEED']}")
                corrected_files = flatten_list(files)
                aws_connector.print_log(request_id, INSTANCE_IDENTIFIER, f"Files being sent to the handler: {corrected_files}", level='INFO')
                return corrected_files
            except Exception as e:
                if attempt_number < MAX_WORKER_ATTEMPTS:
                    aws_connector.print_log(request_id, INSTANCE_IDENTIFIER, f"Worker failed on attempt #{attempt_number}/{MAX_WORKER_ATTEMPTS}. Killing ComfyUI and retrying. Workflow: {payload_template_key}. Exception: {e}", level='WARNING')
                    message_to_log = f"DISTILLERYPRINT: WARNING: Worker failed on attempt #{attempt_number}/{MAX_WORKER_ATTEMPTS}. Killing ComfyUI and retrying. Workflow: {payload_template_key}. Template inputs: {template_inputs}. Exception: {e}"
                    print(message_to_log)
                    time.sleep(0.25)
                    if comfy_connector:
                        comfy_connector.kill_api()
                    attempt_number += 1
                else:
                    message_to_log = f"DISTILLERYPRINT: ERROR: Worker failed on attempt #{attempt_number}/{MAX_WORKER_ATTEMPTS}. Killing ComfyUI and returning None. Workflow: {payload_template_key}. Template inputs: {template_inputs}. Exception: {e}"
                    complete_errorlog = send_runpod_errorlog(message_to_log, request_id)
                    if comfy_connector:
                        comfy_connector.kill_api()
                    return complete_errorlog
    except Exception as e:
        message_to_log = f"DISTILLERYPRINT: ERROR: Unhandled error on worker_routine. Workflow: {payload_template_key}. Exception: {e}"
        complete_errorlog = send_runpod_errorlog(message_to_log, request_id)
        if payload['request_type'] != 'distill':
            if comfy_connector:
                comfy_connector.kill_api()
        return complete_errorlog

def handler(event):
    request_id = 'N/A'
    future = None
    try:
        payload = event['input']
        print(f"payload: {payload}")
        request_id = payload['request_id']
        worker_timeout = WORKER_TIMEOUT_FOR_INFERENCE
        work_assignment = 'INFERENCE'
        if payload['request_type'] == 'distill':
            worker_timeout = WORKER_TIMEOUT_FOR_TRAINING
            work_assignment = 'TRAINING'
        aws_connector = AWSConnector()
        aws_connector.print_log(request_id, INSTANCE_IDENTIFIER, f"DISTILLERYPRINT: Worker called by Master for {work_assignment}. event = {event}.", level='INFO')        
        print(f"DISTILLERYPRINT: Worker called by Master. event = {event}.")
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(worker_routine, event)
            try:
                # Waiting for the result within WORKER_TIMEOUT seconds
                result = future.result(timeout=worker_timeout)
            except TimeoutError:
                # If the timeout occurs, log an error and return a timeout response
                aws_connector.print_log(request_id, INSTANCE_IDENTIFIER, f"DISTILLERYPRINT: Handler timed out after {worker_timeout} seconds doing {work_assignment}.", level='ERROR')
                return None
            aws_connector.print_log(request_id, INSTANCE_IDENTIFIER, f"DISTILLERYPRINT: Worker finished! Throughput time: {(time.time()-START_TIME):.2f} seconds. Work done: {work_assignment}, results: {result}.", level='INFO')
            return result
    except Exception as e:
        complete_errorlog = send_runpod_errorlog("ERROR in Handler", request_id)
        return complete_errorlog
    finally:
        if future:
            future.cancel()
        confirm_disk_space()

runpod.serverless.start({"handler": handler})