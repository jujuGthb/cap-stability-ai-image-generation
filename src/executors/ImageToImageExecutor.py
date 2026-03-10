"""
Stability AI Image Generation Executor: Transforms an existing image guided by a text prompt.
"""

import os
import sys
import cv2
import numpy as np
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.StabilityAI.src.utils.response import build_response_image_to_image
from components.StabilityAI.src.models.PackageModel import PackageModel

API_HOST = "https://api.stability.ai"
ENDPOINTS = {
    "core": "/v2beta/stable-image/generate/core",
    "ultra": "/v2beta/stable-image/generate/ultra",
    "sd3": "/v2beta/stable-image/generate/sd3",
}


class ImageToImageExecutor(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.prompt = self.request.get_param("inputPrompt")
        self.negative_prompt = self.request.get_param("negativePrompt")
        self.strength = self.request.get_param("strength")
        self.model = self.request.get_param("inputModel")
        self.api_key = self.request.get_param("inputApiKey")
        self.image_selector = self.request.get_param("inputImage")
        print(f"[DEBUG] api_key received: '{self.api_key}'")
        print(f"[DEBUG] model: '{self.model}'")
        print(f"[DEBUG] prompt: '{self.prompt}'")
        print(f"[DEBUG] strength: '{self.strength}'")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _build_payload(self):
        data = {
            "prompt": self.prompt,
            "output_format": "jpeg",
            "strength": self.strength if self.strength is not None else 0.3,
        }
        if self.negative_prompt:
            data["negative_prompt"] = self.negative_prompt
        return data

    def run(self):
        img = Image.get_frame(img=self.image_selector, redis_db=self.redis_db)

        success, encoded_image = cv2.imencode('.jpg', img.value)
        if not success:
            raise RuntimeError("Failed to encode image")

        image_bytes = encoded_image.tobytes()

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
                files={"image": ("image.jpg", image_bytes, "image/jpeg")},
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

        return build_response_image_to_image(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()