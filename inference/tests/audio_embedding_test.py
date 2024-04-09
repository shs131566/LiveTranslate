import numpy as np
import tritonclient.http as httpclient

triton_client = httpclient.InferenceServerClient(url="localhost:8000")


model_metadata = triton_client.get_model_metadata("audio_embedding", "1")

audio = np.random.randn(16000 * 5).astype(np.float32)

triton_client.infer(
    model_name="audio_embedding",
    model_version="1",
    inputs=[],
    outputs=[httpclient.InferRequestedOutput("EMBEDDING")],
)
