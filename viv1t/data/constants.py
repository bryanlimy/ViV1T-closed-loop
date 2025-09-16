from pathlib import Path

METADATA_DIR = Path(__file__).parent / "metadata"
STATISTICS_DIR = METADATA_DIR / "statistics"

# mapping of Sensorium 2023 tier names
TIERS = {
    "train": "train",
    "oracle": "validation",
    "live_test_main": "live_main",
    "live_test_bonus": "live_bonus",
    "final_test_main": "final_main",
    "final_test_bonus": "final_bonus",
}

# mouse ID and their corresponding directory relative to data_dir
MOUSE_IDS = {
    # Sensorium 2023 old mice
    "A": "sensorium/dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce",
    "B": "sensorium/dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce",
    "C": "sensorium/dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce",
    "D": "sensorium/dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce",
    "E": "sensorium/dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce",
    # Sensorium 2023 new mice
    "F": "sensorium/dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20",
    "G": "sensorium/dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20",
    "H": "sensorium/dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20",
    "I": "sensorium/dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20",
    "J": "sensorium/dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20",
    # Rochefort Lab
    "K": "rochefort-lab/VIPcre232_FOV1_day1",
    "K_day2": "rochefort-lab/VIPcre232_FOV1_day2",
    "L": "rochefort-lab/VIPcre232_FOV2_day1",
    "L_day2": "rochefort-lab/VIPcre232_FOV2_day2",
    "L_day3": "rochefort-lab/VIPcre232_FOV2_day3",
    "L_day4": "rochefort-lab/VIPcre232_FOV2_day4",
    "M": "rochefort-lab/VIPcre233_FOV1_day1",
    "M_day2": "rochefort-lab/VIPcre233_FOV1_day2",
    "N": "rochefort-lab/VIPcre233_FOV2_day1",
    "N_day2": "rochefort-lab/VIPcre233_FOV2_day2",
    "O": "rochefort-lab/VT288_FOV1_day1",
    "P": "rochefort-lab/VT289_FOV1_day1",
}
MOUSE_DIRS = {v: k for k, v in MOUSE_IDS.items()}

SENSORIUM_OLD = ("A", "B", "C", "D", "E")
SENSORIUM_NEW = ("F", "G", "H", "I", "J")
SENSORIUM = SENSORIUM_OLD + SENSORIUM_NEW
ROCHEFORT_LAB = ("K", "L", "M", "N", "O", "P")

MAX_FRAME = 300  # maximum number of frames in sensorium2023 dataset
FPS = 30

# visual stimulus type
STIMULUS_TYPES = {
    0: "movie",
    1: "directional pink noise",
    2: "gaussian dots",
    3: "random dot kinematogram",
    4: "drifting gabor",
    5: "image",
}
STIMULUS_IDS = {v: k for k, v in STIMULUS_TYPES.items()}

# direction of drifting gabor stimulus in degree
DIRECTIONS = {
    0: "←",
    45: "↙",
    90: "↓",
    135: "↘",
    180: "→",
    225: "↗",
    270: "↑",
    315: "↖",
}

# Monitor information in centimeters (cm)
MONITOR_INFO = {
    mouse_id: {
        "width": 56.5,
        "height": 31.8,
        "distance": 15.0,
    }
    for mouse_id in SENSORIUM
} | {
    mouse_id: {
        "width": 51.0,
        "height": 29.0,
        "distance": 20.0,
    }
    for mouse_id in ROCHEFORT_LAB
}
