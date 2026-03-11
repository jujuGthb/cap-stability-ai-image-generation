from sdks.novavision.src.helper.package import PackageHelper
from sdks.novavision.src.media.image import Image
from components.StabilityAI.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    OutputImage,
    TextToImageExecutor,
    TextToImageResponse,
    TextToImageOutputs,
    ImageToImageExecutor,
    ImageToImageResponse,
    ImageToImageOutputs,
)


def build_response_text_to_image(context):
    output = OutputImage(value=Image.context.image)
    outputs = TextToImageOutputs(outputImage=output)
    response = TextToImageResponse(outputs=outputs)
    executor = TextToImageExecutor(value=response)
    configExecutor = ConfigExecutor(value=executor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)

    return package.build_model(context)


def build_response_image_to_image(context):
    output = OutputImage(value=Image.context.image)
    outputs = ImageToImageOutputs(outputImage=output)
    response = ImageToImageResponse(outputs=outputs)
    executor = ImageToImageExecutor(value=response)
    configExecutor = ConfigExecutor(value=executor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)

    return package.build_model(context)