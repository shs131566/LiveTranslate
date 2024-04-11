import sys

import librosa
import numpy as np
import tritonclient.http as httpclient

SAMPLE_WAV = sys.argv[1]
triton_client = httpclient.InferenceServerClient(
    url="localhost:8000", network_timeout=600, connection_timeout=600
)

model_metadata = triton_client.get_model_metadata("whisper", "1")

audio, sr = librosa.load(SAMPLE_WAV, sr=None)

audio_input = httpclient.InferInput(name="AUDIO", shape=audio.shape, datatype="FP32")
sample_rate_input = httpclient.InferInput(name="SAMPLE_RATE", shape=[1], datatype="INT32")
language_input = httpclient.InferInput(name="LANGUAGE", shape=[1], datatype="BYTES")
inference_type_input = httpclient.InferInput(name="INFERENCE_TYPE", shape=[1], datatype="BYTES")

audio_input.set_data_from_numpy(audio)
sample_rate_input.set_data_from_numpy(np.array([sr], dtype=np.int32))
language_input.set_data_from_numpy(np.array([b"en"]))
inference_type_input.set_data_from_numpy(np.array([b"transcribe"]))

import time

start = time.time()

try:
    result = triton_client.infer(
        model_name="whisper",
        model_version="1",
        inputs=[audio_input, sample_rate_input, language_input, inference_type_input],
    )
except Exception as e:
    print(e)
    print(f"elapsed time: {time.time() - start}")

transcript_result = result.as_numpy("TRANSCRIPT")
print(transcript_result)
repetion_result = result.as_numpy("REPETITION")
print(repetion_result)
language_result = result.as_numpy("LANGUAGE")
print(language_result)
