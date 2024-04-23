# Use an NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install packages in a single layer to reduce image size
RUN apt update && apt upgrade -y && \
    apt-get install -y libgl1 libglib2.0-0 git python3-pip curl && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install -U pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install opencv-python websocket-client watchtower boto3

# Add comfyui
ENV COMFY_UI_REF=d09b5ef4ef150adab31195761725eaba409f6343
RUN  curl -Lsf https://github.com/comfyanonymous/ComfyUI/archive/$COMFY_UI_REF.tar.gz \
    | tar -xz -C / && \
    mv /ComfyUI-$COMFY_UI_REF /ComfyUI

# Add custom nodes
ENV COMFY_UI_MATH_REF=be9beab9923ccf5c5e4132dc1653bcdfa773ed70
RUN curl -Lsf https://github.com/evanspearman/ComfyMath/archive/$COMFY_UI_MATH_REF.tar.gz \
    | tar -xz -C /ComfyUI/custom_nodes && \
    mv /ComfyUI/custom_nodes/ComfyMath-$COMFY_UI_MATH_REF /ComfyUI/custom_nodes/ComfyMath

ENV COMFY_UI_COMFYROLL_REF=d78b780ae43fcf8c6b7c6505e6ffb4584281ceca
RUN curl -Lsf https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/archive/$COMFY_UI_COMFYROLL_REF.tar.gz \
    | tar -xz -C /ComfyUI/custom_nodes && \
    mv /ComfyUI/custom_nodes/ComfyUI_Comfyroll_CustomNodes-$COMFY_UI_COMFYROLL_REF /ComfyUI/custom_nodes/ComfyUI_Comfyroll_CustomNodes

ENV COMFY_UI_CONTROLNET_AUX_REF=c0b33402d9cfdc01c4e0984c26e5aadfae948e05
RUN curl -Lsf https://github.com/Fannovel16/comfyui_controlnet_aux/archive/$COMFY_UI_CONTROLNET_AUX_REF.tar.gz \
    | tar -xz -C /ComfyUI/custom_nodes && \
    mv /ComfyUI/custom_nodes/comfyui_controlnet_aux-$COMFY_UI_CONTROLNET_AUX_REF /ComfyUI/custom_nodes/comfyui_controlnet_aux

ENV COMFY_UI_MANAGER_REF=7c95340ad00339d8a4486c6b9519d5fc2b3fe1cf
RUN curl -Lsf https://github.com/ltdrdata/ComfyUI-Manager/archive/$COMFY_UI_MANAGER_REF.tar.gz \
    | tar -xz -C /ComfyUI/custom_nodes && \
    mv /ComfyUI/custom_nodes/ComfyUI-Manager-$COMFY_UI_MANAGER_REF /ComfyUI/custom_nodes/ComfyUI-Manager

ENV COMFY_UI_STABILITY_REF=001154622564b17223ce0191803c5fff7b87146c
RUN curl -Lsf https://github.com/Stability-AI/stability-ComfyUI-nodes/archive/$COMFY_UI_STABILITY_REF.tar.gz \
    | tar -xz -C /ComfyUI/custom_nodes && \
    mv /ComfyUI/custom_nodes/stability-ComfyUI-nodes-$COMFY_UI_STABILITY_REF /ComfyUI/custom_nodes/stability-ComfyUI-nodes

ENV KOHYA_SS_REF=80091ee701462e0a341ffbb693b9ee81f628d5fd

# We need to patch requirements.txt to:
# 1. Fix path to sd-scripts when installing from /
# 2. Bump opencv to version included in base image
RUN curl -Lsf https://github.com/bmaltais/kohya_ss/archive/$KOHYA_SS_REF.tar.gz \
    | tar -xz -C /ComfyUI && \
    mv /ComfyUI/kohya_ss-$KOHYA_SS_REF /ComfyUI/kohya_ss && \
    sed -i 's|./sd-scripts|ComfyUI/kohya_ss/sd-scripts|' /ComfyUI/kohya_ss/requirements.txt && \
    sed -i 's|opencv-python==.*|opencv-python==4.7.0.72|' /ComfyUI/kohya_ss/requirements.txt

ENV SD_SCRIPTS_REF=bfb352bc433326a77aca3124248331eb60c49e8c
RUN curl -Lsf https://github.com/kohya-ss/sd-scripts/archive/$SD_SCRIPTS_REF.tar.gz \
    | tar -xz -C /ComfyUI/kohya_ss && \
    rmdir /ComfyUI/kohya_ss/sd-scripts && \
    mv /ComfyUI/kohya_ss/sd-scripts-$SD_SCRIPTS_REF /ComfyUI/kohya_ss/sd-scripts

# Install Python libraries from requirements files
RUN pip install \
    -r /ComfyUI/requirements.txt \
    -r /ComfyUI/custom_nodes/ComfyMath/requirements.txt \
    -r /ComfyUI/custom_nodes/comfyui_controlnet_aux/requirements.txt \
    -r /ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt \
    -r /ComfyUI/custom_nodes/stability-ComfyUI-nodes/requirements.txt \
    -r /ComfyUI/kohya_ss/requirements.txt

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Copy the rest of the application
COPY . /app

# Setting the working directory in the container
WORKDIR /workspace

ENV BETTER_EXCEPTIONS=1
ENV API_URL='127.0.0.1'
ENV INITIAL_PORT='8188'
ENV API_COMMAND_LINE='python3 /ComfyUI/main.py --dont-upcast-attention'
ENV APP_NAME='WORKER'
ENV AWS_REGION_NAME='us-east-1'
ENV AWS_LOG_GROUP='xxxxxxxxxxxxxxxxxxxxxxxxxx'
ENV AWS_LOG_STREAM_NAME='xxxxxxxxxxxxxxxxxxxxxxxxxx'
ENV AWS_S3_BUCKET_NAME='xxxxxxxxxxxxxxxxxxxxxxxxxx'
ENV AWS_S3_ACCESS_KEY='xxxxxxxxxxxxxxxxxxxxxxxxxx'
ENV AWS_S3_SECRET_KEY='xxxxxxxxxxxxxxxxxxxxxxxxxx'
ENV AWS_ACCESS_KEY_ID='xxxxxxxxxxxxxxxxxxxxxxxxxx'
ENV AWS_SECRET_ACCESS_KEY='xxxxxxxxxxxxxxxxxxxxxxxxxx'
ENV NETWORK_STORAGE='/runpod-volume'
ENV MODELS_FOLDER='/ComfyUI/models'
ENV CUSTOM_NODES_FOLDER='/ComfyUI/custom_nodes'
ENV WORKER_TIMEOUT_FOR_INFERENCE=360
ENV WORKER_TIMEOUT_FOR_TRAINING=3600
ENV TEST_PAYLOAD='/app/test_payload.json'
ENV REPLICATE_API_TOKEN='xxxxxxxxxxxxxxxxxxxxxxxxxx'
ENV OPENAI_API_KEY='xxxxxxxxxxxxxxxxxxxxxxxxxx'
ENV INFERENCE_OUTPUT_FOLDER='/ComfyUI/output'
ENV INFERENCE_INPUT_FOLDER='/ComfyUI/input'
ENV MINIMUM_GB_FREE_DISK_SPACE=4

# Specifying the command to run the script
ENTRYPOINT ["python3", "-u", "/app/distillery_worker.py"]