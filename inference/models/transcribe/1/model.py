import json
from typing import List

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils import dlpack


class TritonPythonModel:

    def execute(self, requests):
        logger = pb_utils.Logger

        responses: List[pb_utils.InferenceResponse] = []

        for request in requests:
            audio: pb_utils.Tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO")
            sample_rate: pb_utils.Tensor = pb_utils.get_input_tensor_by_name(request, "SAMPLE_RATE")
            language: pb_utils.Tensor = pb_utils.get_input_tensor_by_name(request, "LANGUAGE")
            inference_type: pb_utils.Tensor = pb_utils.Tensor(
                "INFERENCE_TYPE", np.array([b"transcribe"])
            )

            logger.log(
                f"Number of samples: {audio.shape}, sample rate: {sample_rate}, language: {language}",
                logger.INFO,
            )

            # Inference request for whisper model
            whisper_inference_request = pb_utils.InferenceRequest(
                model_name="whisper",
                requested_output_names=["TRANSCRIPT"],
                inputs=[audio, sample_rate, language, inference_type],
            )
            whisper_inference_response = whisper_inference_request.exec()

            if whisper_inference_response.has_error():
                raise pb_utils.TritonModelException(whisper_inference_response.error().message())

            logger.log(f"Transcript: {whisper_inference_response.output_tensors()}", logger.VERBOSE)

            audio_numpy: np.array = audio.as_numpy()
            audio_array = []

            for output_tensor in whisper_inference_response.output_tensors():
                logger.log(output_tensor.as_numpy(), logger.VERBOSE)
                transcript = json.loads(output_tensor.as_numpy()[0].decode("utf-8"))

                text = transcript["text"]
                start = int(float(transcript["start"]) * sample_rate.as_numpy()[0])
                end = int(float(transcript["end"]) * sample_rate.as_numpy()[0])

                audio_array.append(audio[start:end])

            audio_embedding_inference_request = pb_utils.InferenceRequest(
                model_name="audio_embedding",
                requested_output_names=["EMBEDDINGS"],
                inputs=[audio, sample_rate],
            )
            audio_embedding_inference_response = audio_embedding_inference_request.exec()

            if audio_embedding_inference_response.has_error():
                raise pb_utils.TritonModelException(
                    audio_embedding_inference_response.error().message()
                )

            logger.log(
                f"Transcript: {audio_embedding_inference_response.output_tensors().shape}",
                logger.VERBOSE,
            )

        return responses
