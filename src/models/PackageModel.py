from pydantic import validator, Field
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import (
    Package, Image, Inputs, Outputs, Configs, Response, Request, Output, Input, Config
)


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get("value")
        if isinstance(value, list):
            return "list"
        return "object"

    class Config:
        title = "Image"



class OutputImage(Output):
    """
    The generated image produced by Stability AI.
    """
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get("value")
        if isinstance(value, list):
            return "list"
        return "object"

    class Config:
        title = "Output Image"



class InputPrompt(Config):
    """
    Describe what you want to see in the generated image.
    Be as descriptive as possible for better results.
    """
    name: Literal["inputPrompt"] = "inputPrompt"
    value: str = ""
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Prompt"
        json_schema_extra = {"shortDescription": "What you want to see"}


class NegativePrompt(Config):
    """
    Describe what you do NOT want to see in the generated image.
    Example: 'blurry, low quality, distorted, watermark'
    Leave empty if not needed.
    """
    name: Literal["negativePrompt"] = "negativePrompt"
    value: str = ""
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Negative Prompt"
        json_schema_extra = {"shortDescription": "What you don't want to see"}


class InputApiKey(Config):
    """
    Enter your Stability AI API key.
    You can get one at https://platform.stability.ai/account/keys
    """
    name: Literal["inputApiKey"] = "inputApiKey"
    value: str = ""
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "API Key"
        json_schema_extra = {"shortDescription": "Stability AI API Key"}


class ModelCore(Config):
    """
    Balanced speed and quality. Recommended for most use cases.
    """
    name: Literal["core"] = "core"
    value: Literal["core"] = "core"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Core"


class ModelUltra(Config):
    """
    Highest quality output. Slower than Core.
    Best for final renders and high-resolution outputs.
    """
    name: Literal["ultra"] = "ultra"
    value: Literal["ultra"] = "ultra"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Ultra"


class ModelSD3(Config):
    """
    Stable Diffusion 3. More artistic and creative style.
    Best for illustrations and stylized outputs.
    """
    name: Literal["sd3"] = "sd3"
    value: Literal["sd3"] = "sd3"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "SD3"


class InputModel(Config):
    name: Literal["inputModel"] = "inputModel"
    value: Union[ModelCore, ModelUltra, ModelSD3]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Model"
        json_schema_extra = {"shortDescription": "Generation Model"}

class Strength(Config):
    """
    Controls how much the input image influences the generated output.
    Range: 0.0 to 1.0.
    - 0.0: output is nearly identical to the input image.
    - 1.0: input image is completely ignored, generation is from prompt only.
    """
    name: Literal["strength"] = "strength"
    value: float = Field(default=0.3)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Strength"
        json_schema_extra = {
            "shortDescription": "Image Influence (0.0 - 1.0)",
        }

class TextToImageConfigs(Configs):
    inputPrompt: InputPrompt
    negativePrompt: NegativePrompt
    inputModel: InputModel
    inputApiKey: InputApiKey


class TextToImageOutputs(Outputs):
    outputImage: OutputImage


class TextToImageRequest(Request):
    configs: TextToImageConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class TextToImageResponse(Response):
    outputs: TextToImageOutputs


class TextToImageExecutor(Config):
    """
    Generate a new image from a text prompt only.
    No input image required.
    """
    name: Literal["TextToImageExecutor"] = "TextToImageExecutor"
    value: Union[TextToImageRequest, TextToImageResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Text to Image"
        json_schema_extra = {"target": {"value": 0}}


class ImageToImageConfigs(Configs):
    inputPrompt: InputPrompt
    negativePrompt: NegativePrompt
    strength: Strength
    inputModel: InputModel
    inputApiKey: InputApiKey


class ImageToImageInputs(Inputs):
    inputImage: InputImage


class ImageToImageOutputs(Outputs):
    outputImage: OutputImage


class ImageToImageRequest(Request):
    inputs: Optional[ImageToImageInputs]
    configs: ImageToImageConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class ImageToImageResponse(Response):
    outputs: ImageToImageOutputs


class ImageToImageExecutor(Config):
    """
    Generate a new image based on an existing image and a text prompt.
    Use the Strength parameter to control how much the original image is preserved.
    """
    name: Literal["ImageToImageExecutor"] = "ImageToImageExecutor"
    value: Union[ImageToImageRequest, ImageToImageResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Image to Image"
        json_schema_extra = {"target": {"value": 0}}


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[TextToImageExecutor, ImageToImageExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {"shortDescription": "Generation Mode"}


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    name: Literal["StabilityAI"] = "StabilityAI"
    configs: PackageConfigs
    type: Literal["component"] = "component"