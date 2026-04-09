from transformers import pipeline

pipe = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection", device="cpu")
print("Pipeline initialized")
