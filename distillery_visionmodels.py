import base64
import openai
import json
import os
import replicate
import openai

REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')

class VisionModelForCaptioning:

    @staticmethod
    def gptv_caption(image_path):
        prompt_text = """caption this image with super details according to the example\
                that will be provided bellow inside triple dashes ---example---.\
                For the person, only mention man/woman, hair style/color, any facial\
                hair and distinct accessories. Describe the outfit, background, and other\
                details of the image meticulously. Don't talk about emotions and subjective things.\
                \
                ---man, realistic photography, portrait, iphone selfie, looking at viewer,\
                beard, closed mouth, white background, sitting on black office chair,\
                wearing light blue polo shirt---
                Your second task is to reply whether the person on the image is man, woman, or unclear. Just one word.
                The output should be a markdown code snippet formatted in the following schema,\
                including the leading and trailing "```json" and "```":                            
                                ```json
                                {
                                    "image_caption": "{image_caption}",
                                    "man_or_woman": "string"
                                }
                                ```
                """
        def get_image_caption(image_path, prompt_text, model="gpt-4-vision-preview"): # Get caption for an image using GPT-4 with Vision
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt_text
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300
                )
                return response.choices[0]
        try:
            caption_response = get_image_caption(image_path, prompt_text)
            json_str = caption_response.message["content"].replace('```json', '').replace('```', '').strip()
            parsed_json = json.loads(json_str)
            return parsed_json["image_caption"], parsed_json["man_or_woman"]
        except Exception as e:
            raise
    

    @staticmethod
    def llava_caption(image_path):
        LLAVA_CAPTION_PROMPT = """caption this image with super details according to the example\
                                that will be provided bellow inside triple dashes ---example---.\
                                For the person, only mention man/woman, hair style/color, any facial\
                                hair and distinct accessories. Describe the outfit, background, and other\
                                details of the image meticulously. Don't talk about emotions and subjective things.\
                                \
                                ---man, realistic photography, portrait, iphone selfie, looking at viewer,\
                                beard, closed mouth, white background, sitting on black office chair,\
                                wearing light blue polo shirt---
                            """
        LLAVA_GENDER_PROMPT = """Is this a man, woman, or unclear. Answer with only 1 word."""
        def get_llava_response(image_path, prompt, max_tokens=75):
            output = replicate.run(
                "yorickvp/llava-13b:e272157381e2a3bf12df3a8edd1f38d1dbd736bbb7437277c8b34175f8fce358",
                input={
                    "image": open(image_path, "rb"),
                    "prompt": prompt,
                    "top_p": 1,
                    "temperature": 0.25,
                    "max_tokens": max_tokens
                }
            )
            return "".join(output)
        gender_response = get_llava_response(image_path, LLAVA_GENDER_PROMPT, max_tokens=10)
        caption_response = get_llava_response(image_path, LLAVA_CAPTION_PROMPT, max_tokens=75)
        return caption_response, gender_response
