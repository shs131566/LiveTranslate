import json
from typing import Dict, List

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils import dlpack


class TritonPythonModel:
    # Transribe API endpoint is ensemble model of whisper transcribe method and audio_embedding model

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

            transcripts: pb_utils.Tensor = pb_utils.get_output_tensor_by_name(
                whisper_inference_response, "TRANSCRIPT"
            )
            transcripts: np.ndarray = transcripts.as_numpy()
            logger.log(f"transcripts type: {type(transcripts)}", logger.VERBOSE)
            audio_numpy: np.array = audio.as_numpy()

            # Split audio based on whisper transcribe output and send to audio_embedding model
            logger.log(f"Length of transcript: {len(transcripts)}", logger.VERBOSE)

            embeddings = []
            for transcript in transcripts:
                transcript = json.loads(transcript)
                logger.log(f"{transcript}", logger.VERBOSE)

                text = transcript["text"]
                start = int(float(transcript["start"]) * sample_rate.as_numpy()[0])
                end = int(float(transcript["end"]) * sample_rate.as_numpy()[0])

                audio_array: pb_utils.Tensor = pb_utils.Tensor(
                    "AUDIO_ARRAY", np.array([audio_numpy[start:end]])
                )

                audio_embedding_inference_request = pb_utils.InferenceRequest(
                    model_name="audio_embedding",
                    requested_output_names=["EMBEDDINGS"],
                    inputs=[audio_array, sample_rate],
                )
                audio_embedding_inference_response = audio_embedding_inference_request.exec()

                if audio_embedding_inference_response.has_error():
                    raise pb_utils.TritonModelException(
                        audio_embedding_inference_response.error().message()
                    )

                embedding = audio_embedding_inference_response.output_tensors()[0].as_numpy()
                embeddings.append(embedding[0])
            logger.log(f"Number of embeddings1: {len(embeddings)}", logger.INFO)
            # Combine embeddings to clustering model
            embeddings: pb_utils.Tensor = pb_utils.Tensor("EMBEDDINGS", np.vstack(embeddings))
            logger.log(f"Number of embeddings2: {embeddings.shape}", logger.INFO)
            clusters_inference_output = pb_utils.InferenceRequest(
                model_name="clustering", requested_output_names=["SPEAKERS"], inputs=[embeddings]
            ).exec()

            if clusters_inference_output.has_error():
                raise pb_utils.TritonModelException(clusters_inference_output.error().message())

            speakers: pb_utils.Tensor = pb_utils.get_output_tensor_by_name(
                clusters_inference_output, "SPEAKERS"
            )

            transcripts_with_speakers = []
            for idx, speaker in enumerate(speakers.as_numpy()):

                transcript = json.loads(transcripts[idx])
                transcript["speaker"] = speaker
                transcripts_with_speakers.append(transcript)

            transcripts = pb_utils.Tensor(
                "TRANSCRIPT", np.array(transcripts_with_speakers, dtype=np.string_)
            )

            response = pb_utils.InferenceResponse(output_tensors=[transcripts])
            responses.append(response)
        return responses
