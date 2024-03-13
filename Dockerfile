# Using the latest Distillery SD filesystem as the base image
FROM felipeinfante/distillery-worker:base-20240112

# Setting the working directory in the container
WORKDIR /workspace

# Copy the Python script and SD folder into the container
COPY ComfyUI ./ComfyUI
COPY kohya_ss ./kohya_ss
COPY distill ./distill
COPY distillery_aws.py .
COPY distillery_comfy.py .
COPY distillery_worker.py .
COPY distillery_visionmodels.py .
COPY distillery_train.py .
COPY set_env_variables.sh .
COPY docker_run.sh .
COPY test_payload.json .
RUN pip install -U runpod
RUN pip install opencv-python websocket-client watchtower boto3 runpod better-exceptions scikit-image pytz
RUN git config --global --add safe.directory '*'

# Specifying the command to run the script
CMD ["./docker_run.sh"]
