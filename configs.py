from datetime import datetime

def get_config(dataset_name):
    CONFIG = {
        "learning_rate": 1e-3,
        "epoch": 150,
        "C_CODE": 3,      # channel of encode space
        "batches": 1,     # batch size
        "filter": True,   # use gaussian filtering
        "logdir": f"logs/{dataset_name}/" + datetime.now().strftime("%Y%m%d-%H%M%S"),  # result files
        "filters": 50,    # number for convolution kernel
        "crop": 0.2,      # memory limit: [0, 1]
        "dcl_alpha": 10,  # weight for dcl loss term
        "sem_beta": 1     # weight for sem loss term
    }
    return CONFIG