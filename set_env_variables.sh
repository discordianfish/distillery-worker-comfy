#!/bin/bash

export BETTER_EXCEPTIONS=1
export API_URL='127.0.0.1'
export INITIAL_PORT='8188'
export API_COMMAND_LINE='python3 ComfyUI/main.py --dont-upcast-attention'
export APP_NAME='WORKER'
export AWS_REGION_NAME='us-east-1'
export AWS_LOG_GROUP='DistilleryLogGroup'
export AWS_LOG_STREAM_NAME='DistilleryLogStream'
export AWS_S3_BUCKET_NAME='distillerybucket'
export AWS_S3_ACCESS_KEY='AKIA6HX5GHMIWYRZM7UF'
export AWS_S3_SECRET_KEY='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'
export AWS_ACCESS_KEY_ID='AKIAVXEZJ5JJ4JBR42UU'
export AWS_SECRET_ACCESS_KEY='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'
export NETWORK_STORAGE='/runpod-volume'
export MODELS_FOLDER='/workspace/ComfyUI/models'
export CUSTOM_NODES_FOLDER='/workspace/ComfyUI/custom_nodes'
export WORKER_TIMEOUT_FOR_INFERENCE=360
export WORKER_TIMEOUT_FOR_TRAINING=3600
export TEST_PAYLOAD='test_payload.json'
export REPLICATE_API_TOKEN='r8_Y3IAzfBmQzP9YS0lIaLsZ4MI4VsE3nM1h7B2P'
export OPENAI_API_KEY='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'
export INFERENCE_OUTPUT_FOLDER='/workspace/ComfyUI/output'
export INFERENCE_INPUT_FOLDER='/workspace/ComfyUI/input'
export MINIMUM_GB_FREE_DISK_SPACE=4
