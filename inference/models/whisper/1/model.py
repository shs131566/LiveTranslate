import json
import os
from typing import List

import numpy as np
import torch
import torchaudio.transforms as transforms
import triton_python_backend_utils as pb_utils
import whisper
from torch.utils import dlpack

SUPPORT_LANGUGE = ["ko", "en"]


class TritonPythonModel:
    def initialize(self, args):
        logger = pb_utils.Logger

        if args["model_instance_kind"] == "CPU":
            self.device = "cpu"
        elif args["model_instance_kind"] == "GPU":
            self.device = f"cuda:{args['model_instance_device_id']}"

        logger.log(f"device: {self.device}", logger.INFO)

        # Download model if it does not exist
        if not os.path.exists("/models/whisper/1/large-v3.pt"):
            self.model = whisper.load_model(
                "large-v3",
                device=self.device,
            )

        else:
            self.model = whisper.load_model(
                "/models/whisper/1/large-v3.pt",
                device=self.device,
            )
        logger.log("Whisper model large-v3 loaded", logger.INFO)

    def execute(self, requests):
        logger = pb_utils.Logger

        responses: List[pb_utils.InferenceResponse] = []
        for request in requests:
            try:
                # Get input tensors
                audio: pb_utils.Tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO")
                audio: torch.Tensor = dlpack.from_dlpack(audio.to_dlpack())
                logger.log(f"Audio: {audio}", logger.WARNING)
                sample_rate = pb_utils.get_input_tensor_by_name(request, "SAMPLE_RATE").as_numpy()
                language: str = (
                    pb_utils.get_input_tensor_by_name(request, "LANGUAGE")
                    .as_numpy()[0]
                    .decode("utf-8")
                )
                inference_type: str = (
                    pb_utils.get_input_tensor_by_name(request, "INFERENCE_TYPE")
                    .as_numpy()[0]
                    .decode("utf-8")
                )

                logger.log(
                    f"Number of samples: {audio.shape}, sample rate: {sample_rate}, language: {language}, inference type: {inference_type}",
                    logger.INFO,
                )

                # Resample audio to 16kHz if needed, whisper log_mel_spectrogram requires 16kHz
                if sample_rate != 16000:
                    resampler = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    audio = resampler(audio)

                # Detect language if not provided
                if language == "None":
                    audio_tensor = whisper.pad_or_trim(audio).to(self.device)
                    mel = whisper.log_mel_spectrogram(audio_tensor, n_mels=128)

                    language_probs = self.model.detect_language(mel)[1]
                    filtered_probs = {
                        lang: prob
                        for lang, prob in language_probs.items()
                        if lang in SUPPORT_LANGUGE
                    }
                    language = max(filtered_probs, key=filtered_probs.get)

                logger.log(f"Detected language: {language}", logger.INFO)
                options = whisper.DecodingOptions(language=language, without_timestamps=True)

                if inference_type == "transcribe":
                    transcribe_result = self.model.transcribe(
                        audio, language=language, compression_ratio_threshold=1.0
                    )

                    transcripts = []
                    for segment in transcribe_result["segments"]:
                        transcripts.append(
                            {
                                "text": segment["text"],
                                "start": segment["start"],
                                "end": segment["end"],
                            }
                        )

                    repetition = False

                    logger.log(
                        f"Transcript: {transcripts}, repetition: {repetition}", logger.VERBOSE
                    )

                elif inference_type == "streaming":
                    # Pad or trim audio to 30 seconds
                    audio_tensor = whisper.pad_or_trim(audio).to(self.device)
                    mel = whisper.log_mel_spectrogram(audio_tensor, n_mels=128)

                    transcript = {"text": self.model.decode(mel, options)}
                    repetition = transcript.compression_ratio >= 1.0

                else:
                    raise ValueError(f"Invalid inference type: {inference_type}")

                transcript = pb_utils.Tensor("TRANSCRIPT", np.array(transcripts, dtype=np.string_))
                repetition = pb_utils.Tensor("REPETITION", np.array([repetition], dtype=np.bool_))
                language = pb_utils.Tensor(
                    "LANGUAGE", np.array([json.dumps(language)], dtype=np.string_)
                )

                response = pb_utils.InferenceResponse(
                    output_tensors=[transcript, repetition, language]
                )
                responses.append(response)

                logger.log(
                    f"Transcript: {transcript}, repetition: {repetition}, language: {language}",
                    logger.VERBOSE,
                )

            except ValueError as e:
                logger.log(f"Error: {e}", logger.ERROR)
                responses.append(
                    pb_utils.InferenceResponse(
                        f"Triton invalid arg error {e}", error=pb_utils.TritonError.INVALID_ARG
                    )
                )
                continue
            except Exception as e:
                logger.log(f"Error: {e}", logger.ERROR)
                responses.append(
                    pb_utils.InferenceResponse(
                        f"Triton internal error {e}", error=pb_utils.TritonError.INTERNAL
                    )
                )
                continue
        return responses
