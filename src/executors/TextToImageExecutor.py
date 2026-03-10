"""
Stability AI Image Generation Executor: Generates a new image from a text prompt.
"""

import os
import sys
import cv2
import numpy as np
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.StabilityAI.src.utils.response import build_response_text_to_image
from components.StabilityAI.src.models.PackageModel import PackageModel

API_HOST = "https://api.stability.ai"
ENDPOINTS = {
    "core": "/v2beta/stable-image/generate/core",
    "ultra": "/v2beta/stable-image/generate/ultra",
    "sd3": "/v2beta/stable-image/generate/sd3",
}


class TextToImageExecutor(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.prompt = self.request.get_param("inputPrompt")
        self.negative_prompt = self.request.get_param("negativePrompt")
        self.model = self.request.get_param("inputModel")
        self.api_key = self.request.get_param("inputApiKey")
        print(f"[DEBUG] api_key received: '{self.api_key}'")
        print(f"[DEBUG] model: '{self.model}'")
        print(f"[DEBUG] prompt: '{self.prompt}'")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _build_payload(self):
        data = {
            "prompt": self.prompt,
            "output_format": "jpeg",
        }
        if self.negative_prompt:
            data["negative_prompt"] = self.negative_prompt
        return data

    def run(self):
        model = self.model if self.model in ENDPOINTS else "core"
        url = f"{API_HOST}{ENDPOINTS[model]}"

        payload = self._build_payload()

        try:
            response = requests.post(
                url,
                headers={
                    "authorization": f"Bearer {self.api_key}",
                    "accept": "image/*"
                },
                files={"none": ""},
                data=payload
            )

            print(f"[DEBUG] Status: {response.status_code}")

            response.raise_for_status()

            image_array = np.frombuffer(response.content, dtype=np.uint8)
            self.output_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] HTTP Error {response.status_code}: {response.text}")
            self.output_image = None
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.output_image = None

        return build_response_text_to_image(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()