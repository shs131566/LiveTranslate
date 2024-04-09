import json
import os
from typing import List

import huggingface_hub
import torch
import triton_python_backend_utils as pb_utils
from wespeaker.wespeaker_resnet import WeSpeakerResNet34


class TritonPythonModel:
    def initialize(self, args):
        logger = pb_utils.Logger

        self.model_config = model_config = json.loads(args["model_config"])
        embeddings_config = pb_utils.get_output_config_by_name(
            model_config, "EMBEDDING"
        )
        self.embeddings_dtype = pb_utils.triton_string_to_numpy(
            embeddings_config["data_type"]
        )

        if args["model_instance_kind"] == "CPU":
            self.device = "cpu"
        elif args["model_instance_kind"] == "GPU":
            self.device = f"cuda:{args['model_instance_device_id']}"

        logger.log(f"device: {self.device}", logger.INFO)

        # Download model if it does not exist
        if not os.path.exists("/models/audio_embedding/1/pytorch_model.bin"):
            logger.log(
                "Downloading model wespeaker/wespeaker-voxceleb-resnet34-LM",
                logger.INFO,
            )
            huggingface_hub.hf_hub_download(
                repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
                filename="pytorch_model.bin",
                local_dir="/models/audio_embedding/1/",
                force_download=True,
                local_dir_use_symlinks=False,
            )

        model = WeSpeakerResNet34.load_from_checkpoint(
            checkpoint_path="/models/audio_embedding/1/pytorch_model.bin",
            map_location=self.device,
        )
        model.eval()
        model.to(self.device)

        logger.log("Audio embedding model loaded", logger.INFO)

    def execute(self, requests):
        logger = pb_utils.Logger
        embeddings_dtype = self.embeddings_dtype

        responses: List[pb_utils.InferenceResponse] = []

        for request in requests:
            try:
                audio = pb_utils.get_input_tensor_by_name(request, "AUDIO").as_numpy()
                sample_rate = pb_utils.get_input_tensor_by_name(
                    request, "SAMPLE_RATE"
                ).as_numpy()

                logger.info(
                    f"Audio duration: {len(audio)/sample_rate} seconds, sample rate: {sample_rate} Hz"
                )

                audio_tensor = torch.Tensor(audio).reshape(1, 1, -1)
                self.model.sample_rate = sample_rate
                embedding = self.model(audio_tensor)

                logger.log(
                    f"Embedding shape: {embedding.shape}",
                    logger.INFO,
                )

                embedding = pb_utils.Tensor(
                    "EMBEDDING",
                    embedding.astype(embeddings_dtype),
                )
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[embedding]
                )
            except Exception as e:
                logger.log(f"Error: {e}", logger.ERROR)
                inference_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError.INTERNAL
                )

            responses.append(inference_response)

        return responses
