from pathlib import Path
import os

LARGE_DICT = {
    "INFERENCE_WEIGHTS_PATH": str(Path(__file__).parent.parent / "ckpts/weights-large.pth"),
    "NUM_SAMPLES": 1000,
    "TOP_K": 9,
    "TOP_P": 0.9,
    "TEMPERATURE": 1.2,
    "PATCH_STREAM": True,
    "PATCH_SIZE": 16,
    "PATCH_LENGTH": 1024,
    "CHAR_NUM_LAYERS": 6,
    "PATCH_NUM_LAYERS": 20,
    "HIDDEN_SIZE": 1280,
    "LR": 1e-5
}

MEDIUM_DICT = {
    "INFERENCE_WEIGHTS_PATH": str(Path(__file__).parent.parent / "ckpts/weights-medium.pth"),
    "NUM_SAMPLES": 1000,
    "TOP_K": 9,
    "TOP_P": 0.9,
    "TEMPERATURE": 1.2,
    "PATCH_STREAM": True,
    "PATCH_SIZE": 16,
    "PATCH_LENGTH": 2048,
    "CHAR_NUM_LAYERS": 3,
    "PATCH_NUM_LAYERS": 16,
    "HIDDEN_SIZE": 1024,
    "LR": 1e-5
}

SMALL_DICT = {
    "INFERENCE_WEIGHTS_PATH": str(Path(__file__).parent.parent / "ckpts/weights-small.pth"),
    "NUM_SAMPLES": 1000,
    "TOP_K": 1,
    "TOP_P": 0.9,
    "TEMPERATURE": 1.2,
    "PATCH_STREAM": True,
    "PATCH_SIZE": 16,
    "PATCH_LENGTH": 2048,
    "CHAR_NUM_LAYERS": 3,
    "PATCH_NUM_LAYERS": 12,
    "HIDDEN_SIZE": 768,
    "LR": 1e-5
}

def get_notagen_params(model_type: str="large")-> dict:
    assert model_type in ["large", "medium", "small"], \
        f"Model types are 'large', `medium` and `small`."
    
    if model_type == "large":
        res =  LARGE_DICT
    elif model_type == "medium":
        res = MEDIUM_DICT
    else:
        res = SMALL_DICT
    
    INFERENCE_WEIGHTS_PATH = res["INFERENCE_WEIGHTS_PATH"]
    TOP_K = res["TOP_K"]
    TOP_P = res["TOP_P"]
    TEMPERATURE = res["TEMPERATURE"]
    ORIGINAL_OUTPUT_FOLDER = os.path.join('./output/original', \
        os.path.splitext(os.path.split(INFERENCE_WEIGHTS_PATH)[-1])[0] + '_k_' + str(TOP_K) + '_p_' + str(TOP_P) + '_temp_' + str(TEMPERATURE))
    INTERLEAVED_OUTPUT_FOLDER = os.path.join('./output/interleaved', \
        os.path.splitext(os.path.split(INFERENCE_WEIGHTS_PATH)[-1])[0] + '_k_' + str(TOP_K) + '_p_' + str(TOP_P) + '_temp_' + str(TEMPERATURE))

    res["ORIGINAL_OUTPUT_FOLDER"] = ORIGINAL_OUTPUT_FOLDER
    res["INTERLEAVED_OUTPUT_FOLDER"] = INTERLEAVED_OUTPUT_FOLDER

    return res
