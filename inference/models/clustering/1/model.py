from typing import List

import numpy as np
import triton_python_backend_utils as pb_utils
from cluster import AgglomerativeClustering


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger

    def execute(self, requests):
        responses: List[pb_utils.InferenceResponse] = []
        for request in requests:
            try:
                embeddings: pb_utils.Tensor = pb_utils.get_input_tensor_by_name(
                    request, "EMBEDDINGS"
                )
                embeddings = embeddings.as_numpy()
                speakers = []
                self.logger.log(f"Number of embeddings: {embeddings.shape}", self.logger.INFO)
                if len(embeddings) == 1:
                    speakers.append(0)
                else:
                    cluster_model = AgglomerativeClustering()
                    clusters = cluster_model.cluster(np.vstack(embeddings))

                    for cluster in clusters:
                        speakers.append(int(cluster))

                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("SPEAKERS", np.array(speakers, dtype=np.int32))]
                )
                responses.append(inference_response)

            except ValueError as e:
                self.logger.log(f"Error: {e}", self.logger.ERROR)
                responses.append(
                    pb_utils.InferenceResponse(
                        f"Triton invalid arg error {e}", error=pb_utils.TritonError.INVALID_ARG
                    )
                )
                continue
            except Exception as e:
                self.logger.log(f"Error: {e}", self.logger.ERROR)
                responses.append(
                    pb_utils.InferenceResponse(
                        f"Triton internal error {e}", error=pb_utils.TritonError.INTERNAL
                    )
                )
                continue
        return responses
