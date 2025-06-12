from earth2mip.networks import get_model

MODEL_PATH = "/home/jovyan/fcnv2_sm"
model = get_model(f"file://{MODEL_PATH}")
print("Successfully obtained model!")