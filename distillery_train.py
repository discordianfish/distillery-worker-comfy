import os
import shutil
from distillery_visionmodels import VisionModelForCaptioning
from distillery_aws import AWSConnector
import subprocess
import shlex
from PIL import Image
from io import BytesIO

WORKSPACE_FOLDER = '/workspace'
BASE_TRAINING_FOLDER = 'distill'
TEMP_FOLDER_NAME = 'temp_img'
NETWORK_STORAGE = os.getenv("NETWORK_STORAGE") # Path to network storage mount
MODELS_FOLDER = os.getenv("MODELS_FOLDER") # Path to models folder in ComfyUI
BASE_MODEL = "Cosmopolitan_release_version.safetensors"

class TrainingSetup:
    @classmethod
    def step1_create_project_folders(cls, lora_name): # Create the project folder and subfolders, and download the image file from S3 to the project folder
        try:
            temp_folder = f"{WORKSPACE_FOLDER}/{BASE_TRAINING_FOLDER}/{TEMP_FOLDER_NAME}" # first, create the temp folder if it doesn't exist
            if not os.path.exists(temp_folder): 
                os.makedirs(temp_folder)
            project_folder = f"{WORKSPACE_FOLDER}/{BASE_TRAINING_FOLDER}/{lora_name}" # Now, create the project folder and subfolders
            subfolders = ['img', 'log', 'model', 'reg']
            if not os.path.exists(project_folder):
                os.makedirs(project_folder)
                for folder in subfolders:
                    os.makedirs(os.path.join(project_folder, folder))
            else:
                raise FileExistsError(f"The project folder '{lora_name}' already exists.")
            return temp_folder, project_folder # temp_folder is where the input image will be downloaded to; project_folder is where the training will happen
        except Exception as e:
            raise

    @classmethod
    def step2_download_image_file(cls, lora_name, temp_folder, original_image_file_name):
        try:
            aws_connector = AWSConnector()
            _, file_extension = os.path.splitext(original_image_file_name)
            image_file_name = f"{lora_name}{file_extension}" # renaming the image file to the lora name
            aws_connector.download_files([(original_image_file_name, f"{temp_folder}/{image_file_name}")]) # setting the input image path to a temporary project folder and renaming it to the lora name    
            image_file_path = f"{temp_folder}/{image_file_name}"
            return image_file_name, image_file_path
        except Exception as e:
            raise

    @classmethod
    def step3_caption_image(cls, input_image_file_folder, force_llava = False, force_category=None): # Will try OpenAI API first
        def define_category(value, force_category=None):
            try:
                if force_category is None:
                    normalized_value = value.lower() # Convert the value to lowercase for standardization
                    if 'woman' in normalized_value: # Check and return the normalized value
                        return 'woman'
                    elif 'man' in normalized_value:
                        return 'man'
                    else:
                        return 'person'
                else:
                    return force_category
            except Exception as e:
                raise
        try:
            image_caption = None
            subject_category = None # currently accepts 'woman', 'man', or 'person'
            if not force_llava:
                try:
                    image_caption, subject_category = VisionModelForCaptioning.gptv_caption(input_image_file_folder)
                except Exception as e:
                    image_caption, subject_category = VisionModelForCaptioning.llava_caption(input_image_file_folder) # Fallback to Replicate API if OpenAI API fails
            if image_caption is None or subject_category is None:
                image_caption, subject_category = VisionModelForCaptioning.llava_caption(input_image_file_folder) # Fallback to Replicate API if OpenAI API fails
            subject_category = define_category(subject_category, force_category=force_category)
            return image_caption, subject_category
        except Exception as e:
            raise

    @classmethod
    def step4_setup_regularization_images(cls, project_folder, subject_category, num_repetitions=1):
        def copy_images(source_folder, destination_folder):
            try:
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                for file_name in os.listdir(source_folder):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.npz')):
                        shutil.copy(os.path.join(source_folder, file_name), destination_folder)
            except Exception as e:
                raise
        try:
            source_folder = f"{WORKSPACE_FOLDER}/{BASE_TRAINING_FOLDER}/reg/{subject_category}"
            new_folder_path = f"{project_folder}/reg/{num_repetitions}_{subject_category}"
            # Create new folder and copy images
            copy_images(source_folder, new_folder_path)
        except Exception as e:
            raise

    @classmethod
    def step5_prepare_training_setup(cls, lora_name, project_folder, image_file_name, image_file_path, image_caption, subject_category, num_reg_images=200):
        try:
            # First, create the required folder where the training will happen
            new_folder_path = f"{project_folder}/img/{num_reg_images}_{lora_name}_{subject_category}"
            os.makedirs(new_folder_path, exist_ok=True)
            # Then, copy the image to the new folder
            new_image_path = f'{new_folder_path}/{image_file_name}'
            shutil.copy(image_file_path, new_image_path)
            # Finally, create a text file with the image caption
            with open(f"{new_folder_path}/{lora_name}.txt", 'w') as text_file:
                text_file_content = f"{lora_name}, {image_caption}"
                text_file.write(text_file_content)
        except Exception as e:
            raise

    @classmethod
    def do_setup(cls, lora_name, original_image_file_name, force_category=None):
        temp_folder, project_folder = cls.step1_create_project_folders(lora_name) # Create the project folder and subfolders
        image_file_name, image_file_path = cls.step2_download_image_file(lora_name, temp_folder, original_image_file_name) # Download the image file from S3 to the project folder
        image_caption, subject_category = cls.step3_caption_image(image_file_path, force_category=force_category) # Caption the image
        cls.step4_setup_regularization_images(project_folder, subject_category) # Setup the regularization images folder
        cls.step5_prepare_training_setup(lora_name, project_folder, image_file_name, image_file_path, image_caption, subject_category) # Create the required folder and copy the image to the new folder
        return project_folder, image_caption, subject_category, image_file_path

class TrainingExecution:
    @classmethod
    def run_training_algorithm(cls, lora_name, original_image_file_name, force_category=None):
        try:
            project_folder, image_caption, subject_category, image_file_path = TrainingSetup.do_setup(lora_name, original_image_file_name, force_category=force_category)
            full_path_to_base_model = f"{MODELS_FOLDER}/checkpoints/{BASE_MODEL}"            
            training_command = f'accelerate launch --num_cpu_threads_per_process=2 {WORKSPACE_FOLDER}/kohya_ss/train_network.py --enable_bucket --min_bucket_reso=256 --max_bucket_reso=2048 --pretrained_model_name_or_path="{full_path_to_base_model}" --train_data_dir="{project_folder}/img" --reg_data_dir="{project_folder}/reg" --resolution="768,768" --output_dir="{project_folder}/model" --logging_dir="{project_folder}/log" --network_alpha="1" --save_model_as=safetensors --network_module=lycoris.kohya --network_args "conv_dim=1" "conv_alpha=1" "use_cp=False" "algo=loha" --network_dropout="0" --text_encoder_lr=1.0 --unet_lr=1.0 --network_dim=128 --output_name="{lora_name}" --lr_scheduler_num_cycles="3" --scale_weight_norms="1" --no_half_vae --learning_rate="1.0" --lr_scheduler="cosine" --train_batch_size="8" --max_train_steps="100" --save_every_n_epochs="3" --mixed_precision="bf16" --save_precision="bf16" --seed="1991" --caption_extension=".txt" --cache_latents --cache_latents_to_disk --optimizer_type="DAdaptAdam" --optimizer_args decouple=True use_bias_correction=True weight_decay=0.20 --keep_tokens="2" --bucket_reso_steps=64 --min_snr_gamma=5 --flip_aug --shuffle_caption --gradient_checkpointing --xformers --bucket_no_upscale --noise_offset=0.0375'
            print(f"DISTILLERYPRINT - TRAINING COMMAND: {training_command}")
            args = shlex.split(training_command) # Splitting the command into a list of arguments
            subprocess.run(args) # Executing the command
            return project_folder, image_caption, subject_category, image_file_path
        except Exception as e:
            raise

    @classmethod
    def save_and_upload_model(cls, project_folder, lora_name, image_file_path):
        def convert_to_png_object(image_path, output_key):
            try:
                image = Image.open(image_path) # Read the image
                in_memory_file = BytesIO() # Convert the image to PNG and store it in a BytesIO object
                image.save(in_memory_file, format='PNG')
                return in_memory_file
            except Exception as e:
                raise
        try:
            # Save the model to S3
            aws_connector = AWSConnector()
            model_file_name = f"{lora_name}.safetensors"
            model_file_full_path = f"{project_folder}/model/{model_file_name}"
            aws_connector.upload_files([(model_file_full_path, model_file_name)])
            # Upload the image to S3 converted to PNG for standardization
            base_image_file_name = f"{lora_name}.png"
            in_memory_file = convert_to_png_object(image_file_path, base_image_file_name)
            aws_connector.upload_fileobj([(in_memory_file, f"{lora_name}.png")]) # Upload the file
            # Save the model to the Runpod database => this should only be used if the trainer worker has access to the Runpod network storage
            subprocess.run(["cp", model_file_full_path, f"{NETWORK_STORAGE}/loras/"], check=True)
            # Move the image to the ComfyUI path => this should only be used if the trainer worker also does the inferencing
            subprocess.run(["mv", model_file_full_path, f"{MODELS_FOLDER}/loras/{model_file_name}"], check=True)
            return model_file_name, base_image_file_name
        except Exception as e:
            raise

def do_training(lora_name, original_image_file_name, force_category=None):
    try:
        project_folder, image_caption, subject_category, image_file_path = TrainingExecution.run_training_algorithm(lora_name, original_image_file_name, force_category=force_category)
        model_file_name, base_image_file_name = TrainingExecution.save_and_upload_model(project_folder, lora_name, image_file_path)
        # Prepare the output
        output = {}
        output['lora_name'] = lora_name
        output['lora_model_file_name'] = model_file_name
        output['image_file'] = base_image_file_name
        output['image_caption'] = image_caption
        output['subject_category'] = subject_category
        return output
    except Exception as e:
        raise
    finally:
        try:
            shutil.rmtree(project_folder)
        except Exception as e:
            pass # Ignore if the folder doesn't exist