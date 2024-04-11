import numpy as np
import tritonclient.http as httpclient

triton_client = httpclient.InferenceServerClient(url="localhost:8000")


model_metadata = triton_client.get_model_metadata("audio_embedding")

# Generate random audio data
audio = np.random.randn(16000 * 5).astype(np.float32)
# Copy the audio data 100 times to create a batch of size 100
audio_array = np.tile(audio, (100, 1))

audio_input = httpclient.InferInput(name="AUDIO_ARRAY", shape=audio_array.shape, datatype="FP32")
sample_rate_input = httpclient.InferInput(name="SAMPLE_RATE", shape=[1], datatype="INT32")

audio_input.set_data_from_numpy(audio_array)
sample_rate_input.set_data_from_numpy(np.array([16000], dtype=np.int32))

result = triton_client.infer(
    model_name="audio_embedding",
    model_version="1",
    inputs=[audio_input, sample_rate_input],
)
embedding_result = result.as_numpy("EMBEDDINGS")
print(embedding_result.shape)
