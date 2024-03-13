import boto3
from watchtower import CloudWatchLogHandler
from typing import List, Tuple
import os
import logging
import inspect
import time
from io import BytesIO
import json
import time
import socket
import copy
import datetime
import pytz

APP_NAME = os.getenv('APP_NAME')
AWS_REGION_NAME = os.getenv('AWS_REGION_NAME')
AWS_LOG_GROUP = os.getenv('AWS_LOG_GROUP')
AWS_LOG_STREAM_NAME = os.getenv('AWS_LOG_STREAM_NAME')
AWS_S3_BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')
AWS_S3_ACCESS_KEY = os.getenv('AWS_S3_ACCESS_KEY')
AWS_S3_SECRET_KEY = os.getenv('AWS_S3_SECRET_KEY')

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = record.__dict__.copy()
        # Check if 'msg' is already in JSON format
        try:
            log_dict = json.loads(log_record['msg'])
        except Exception as e:
            log_dict = copy.deepcopy(log_record['msg'])
        log_dict['level'] = record.levelname
        # return the JSON string
        return json.dumps(log_dict)

class AWSConnector:
    _instance = None

    def __new__(cls):
        try:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance.region_name = AWS_REGION_NAME
                cls._instance.log_group = AWS_LOG_GROUP
                cls._instance.log_stream_name = AWS_LOG_STREAM_NAME
                cls._instance.setup_logging()
            return cls._instance
        except Exception as e:
            raise
    
    def setup_logging(self, level=logging.INFO): 
        try:
            app_logger = logging.getLogger(APP_NAME)
            if not app_logger.hasHandlers():  # Check if handlers are already added
                app_logger.setLevel(level)
                # Setup CloudWatch log handler
                session = boto3.Session(region_name=self.region_name)
                cloudwatch_client = session.client('logs')
                cw_handler = CloudWatchLogHandler(boto3_client=cloudwatch_client, log_group=self.log_group, stream_name=self.log_stream_name)
                json_formatter = JSONFormatter()
                cw_handler.setFormatter(json_formatter)
                app_logger.addHandler(cw_handler)
                app_logger.propagate = False
        except Exception as e:
            raise

    def print_log(self, request_id, context, message, level='INFO'): 
        try:
            app_logger = logging.getLogger(APP_NAME)
            caller_frame = inspect.currentframe().f_back
            script_name = os.path.basename(caller_frame.f_globals["__file__"])
            line_number = caller_frame.f_lineno
            function_name = caller_frame.f_code.co_name
            hostname = socket.gethostname()
            # Convert timestamp to EST
            unixtime = time.time()
            utc_time = datetime.datetime.utcfromtimestamp(unixtime)
            est_time = utc_time.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('US/Eastern'))
            log_data = {
                "context": context,
                "esttime": est_time.strftime("%Y-%m-%d %H:%M:%S"),  # EST timestamp
                "unixtime": unixtime,  # Unix timestamp
                "request_id": request_id,
                "message": message,
                "script_name": script_name,
                "function_name": function_name,
                "line_number": line_number,
                "hostname": f"{APP_NAME}-{hostname}"
            }
            message_to_print = json.dumps(log_data)
            if level == 'INFO':
                app_logger.info(message_to_print)
            elif level == 'ERROR':
                app_logger.error(message_to_print)
            elif level == 'WARNING':
                app_logger.warning(message_to_print)
        except Exception as e:
            raise

    def upload_fileobj(self, files: List[Tuple[BytesIO, str]]):
        try:
            s3 = boto3.client('s3', aws_access_key_id=AWS_S3_ACCESS_KEY, aws_secret_access_key=AWS_S3_SECRET_KEY, region_name=AWS_REGION_NAME)
            for file_obj, key in files:
                file_obj.seek(0)  # Ensure we're at the start of the file
                s3.upload_fileobj(file_obj, AWS_S3_BUCKET_NAME, key)
        except Exception as e:
            raise

    def download_fileobj(self, keys: List[str]) -> List[BytesIO]:
        try:
            file_objs = []
            s3 = boto3.client('s3', aws_access_key_id=AWS_S3_ACCESS_KEY, aws_secret_access_key=AWS_S3_SECRET_KEY, region_name=AWS_REGION_NAME)
            for key in keys:
                file_obj = BytesIO()
                s3.download_fileobj(AWS_S3_BUCKET_NAME, key, file_obj)
                file_obj.seek(0)  # Ensure we're at the start of the file
                file_objs.append(file_obj)
                return file_objs
        except Exception as e:
            raise

    def upload_files(self, files: List[Tuple[str, str]]):
        try:
            s3 = boto3.client('s3', aws_access_key_id=AWS_S3_ACCESS_KEY, aws_secret_access_key=AWS_S3_SECRET_KEY, region_name=AWS_REGION_NAME)
            for file_name, key in files:
                s3.upload_file(file_name, AWS_S3_BUCKET_NAME, key)
        except Exception as e:
            raise

    def download_files(self, files: List[Tuple[str, str]]):
        try:
            s3 = boto3.client('s3', aws_access_key_id=AWS_S3_ACCESS_KEY, aws_secret_access_key=AWS_S3_SECRET_KEY, region_name=AWS_REGION_NAME)
            for key, file_name in files:
                s3.download_file(AWS_S3_BUCKET_NAME, key, file_name)
        except Exception as e:
            raise
