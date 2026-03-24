import sherpa_onnx
import os

MODEL_PATH = r"models/zipformer"
tokens_path = os.path.join(MODEL_PATH, "tokens.txt")
encoder_path = os.path.join(MODEL_PATH, "encoder-epoch-99-avg-1.int8.onnx")
decoder_path = os.path.join(MODEL_PATH, "decoder-epoch-99-avg-1.onnx")
joiner_path = os.path.join(MODEL_PATH, "joiner-epoch-99-avg-1.int8.onnx")

recog = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=tokens_path,
    encoder=encoder_path,
    decoder=decoder_path,
    joiner=joiner_path,
    decoding_method="greedy_search",
    sample_rate=16000,
    feature_dim=80,
)

print("Recognizer loaded successfully!")
