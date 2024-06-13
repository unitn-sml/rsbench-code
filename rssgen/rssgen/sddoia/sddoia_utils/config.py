# Module which contains all the configurable of SDDOIA

import yaml


def load_colors_from_yaml(file_path):
    """Load colors"""
    with open(file_path, "r") as file:
        colors_data = yaml.safe_load(file)
        return colors_data["COLORS"]


def load_config_from_yaml(file_path):
    """Load YAML Config"""
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
        return config_data["CONFIG"]


COLORS = load_colors_from_yaml("../../boia_config/colors.yml")

# Convert RGB colors to RGBA
COLORS_TO_RGBA = {
    name: [c / 255.0 for c in rgb] + [1.0] for name, rgb in COLORS.items()
}

# Batch of materials
MATERIALS = {"rubber": "Rubber", "metal": "MyMetal"}

# CONFIG
CONFIG = load_config_from_yaml("../../boia_config/config.yml")

# Constants defining the models
CAR_MODEL = "../../boia_config/shapes/car_new.blend"
CARS_COLORS = [
    "../../boia_config/shapes/car_new.blend",
    "../../boia_config/shapes/car_new_blue.blend",
    "../../boia_config/shapes/car_new_brown.blend",
    "../../boia_config/shapes/car_new_cyan.blend",
    "../../boia_config/shapes/car_new_green.blend",
    "../../boia_config/shapes/car_new_purple.blend",
    "../../boia_config/shapes/car_new_white.blend",
    "../../boia_config/shapes/car_new_yellow.blend",
]
MPH10 = "../../boia_config/shapes/10.blend"
MPH20 = "../../boia_config/shapes/20.blend"
MPH30 = "../../boia_config/shapes/30.blend"
MPH40 = "../../boia_config/shapes/40.blend"
MPH50 = "../../boia_config/shapes/50.blend"
MPH60 = "../../boia_config/shapes/60.blend"
MPH70 = "../../boia_config/shapes/70.blend"
MPH80 = "../../boia_config/shapes/80.blend"
MPH90 = "../../boia_config/shapes/90.blend"
MPH100 = "../../boia_config/shapes/10.blend"
BARRIER = "../../boia_config/shapes/barrier.blend"
RIDER_MODEL = "../../boia_config/shapes/Rider.blend"
PEDESTRIAN_MODEL = "../../boia_config/shapes/Stickman.blend"
STOP = "../../boia_config/shapes/stopsign.blend"
TLG = "../../boia_config/shapes/TL_G.blend"
TLY = "../../boia_config/shapes/TL_Y.blend"
TLR = "../../boia_config/shapes/TL_R.blend"
TREE_MODEL = "../../boia_config/shapes/Tree.blend"
LINE = "../../boia_config/shapes/Line.blend"
DASHED_LINE = "../../boia_config/shapes/DashedLine.blend"
CROSS_WALK_MODEL = "../../boia_config/shapes/CrossWalk.blend"
STREET_CONE = "../../boia_config/shapes/StreetCone.blend"


# CONSTANTS defining the scenes
BOIA_FULL = "../../boia_config/base_scene_boia.blend"
BOIA_LEFT = "../../boia_config/base_scene_boia_left_only.blend"
BOIA_RIGHT = "../../boia_config/base_scene_boia_right_only.blend"
BOIA_STRAIGHT = "../../boia_config/base_scene_boia_single_road.blend"

# Models
SEM_MODELS = {0: None, 1: TLG, 2: TLY, 3: TLR}
SEM_NAME = {TLG: "TLG", TLY: "TLY", TLR: "TLR"}
STOP_SIGN_MODELS = {0: None, 1: STOP}
BASE_DIM = [1.7, 3.2, 1.2]

# Define categorical distributions
TL_DIST = {"": 0.3, "G": 0.4, "R": 0.2, "Y": 0.1}
TL_DIST_FORW = {"": 0.6, "G": 0.4}
TL_DIST_LEFT = {"": 0.3, "G": 0.4}
TL_DIST_RIGHT = {"": 0.3, "G": 0.4}
OBS_DIST = {"car-same-dir": 0.1, "car-diff-dir": 0.2, "": 0.1, "misc": 0.2}
OBS_DIST = {
    "values": [
        ["car-same-dir"],
        ["car-diff-dir"],
        ["pedestrian"],
        ["rider"],
        ["other"],
        [""],
        ["pedestrian", "rider"],
        ["pedestrian", "rider", "car-same-dir"],
        ["pedestrian", "rider", "car-diff-dir"],
        ["pedestrian", "rider", "other"],
        ["pedestrian", "other"],
        ["pedestrian", "car-same-dir"],
        ["pedestrian", "car-diff-dir"],
        ["rider", "other"],
        ["rider", "car-same-dir"],
        ["rider", "car-diff-dir"],
    ],
    "weights": [
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.1,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
    ],
}
OBS_DIST_FORW = {
    "values": [
        ["car-same-dir"],
        [""],
    ],
    "weights": [0.5, 0.5],
}
OBS_DIST_LEFT = {
    "values": [
        ["left-follow"],
        [""],
    ],
    "weights": [0.50, 0.50],
}

OBS_DIST_RIGHT = {
    "values": [
        ["right-follow"],
        [""],
    ],
    "weights": [0.50, 0.50],
}

SIGN_DIST = {"": 0.7, "nonstop": 0.2, "stop": 0.1}
SIGN_DIST_FORW = {"": 0.7, "nonstop": 0.3}

# Relevant positions
RELEVANT_SEMAPHORE_POSITION = [(1, 4)]
RELEVANT_SIGN_POSITION = [(1, 4)]
RELEVANT_PEDESTRIAN_POSITION = [(2, 2), (2, 3), (3, 2), (3, 3)]
RELEVANT_CAR_POSITION = [(1, 3), (2, 3), (3, 3), (4, 3), (5, 3)]
RELEVANT_OTHER_POSITION = [(1, 3), (2, 3), (3, 3), (4, 3), (5, 3)]

# Car position matrix
CAR_POSITION_MATRIX = {
    (0, 3): {
        "camera": {"object": "Empty.001", "direction": "forward"},
        "car": {"object": "Cube.015", "direction": "forward", "blend": CAR_MODEL},
        "obstacles": {
            "car": [
                {
                    "object": "Cube.015",
                    "semantic": "Car",
                    "blend": CAR_MODEL,
                    "direction": ["forward"],
                }
            ],
        },
    },
    (1, 3): {
        "camera": {"object": "Empty.003", "direction": "forward"},
        "car": {"object": "Cube.012", "direction": "forward", "blend": CAR_MODEL},
        "obstacles": {
            "car": [
                {
                    "object": "Cube.012",
                    "semantic": "Car",
                    "blend": CAR_MODEL,
                    "direction": ["left", "right", "forward", "backward"],
                }
            ],
            "other": [
                {
                    "object": "Suzanne",
                    "semantic": "StreetCone",
                    "blend": STREET_CONE,
                    "direction": ["left", "right"],
                },
                {
                    "object": "Suzanne",
                    "semantic": "Barrier",
                    "blend": BARRIER,
                    "direction": ["left", "right"],
                },
            ],
        },
    },
    (2, 2): {
        "obstacles": {
            "pedestrian": [
                {
                    "object": "Cylinder.003",
                    "semantic": "Stickman",
                    "blend": PEDESTRIAN_MODEL,
                    "direction": ["left", "right"],
                }
            ],
            "rider": [
                {
                    "object": "Cylinder.003",
                    "semantic": "Rider",
                    "blend": RIDER_MODEL,
                    "direction": ["left", "right"],
                }
            ],
            "car": [
                {
                    "object": "Cube.016",
                    "semantic": "Car",
                    "blend": CAR_MODEL,
                    "direction": ["left", "right", "forward", "backward"],
                }
            ],
        }
    },
    (2, 3): {
        "camera": {"object": "Empty.004", "direction": "forward"},
        "car": {
            "object": "Cube.019",
            "direction": ["forward", "left", "right"],
            "blend": CAR_MODEL,
        },
        "obstacles": {
            "pedestrian": [
                {
                    "object": "Cylinder.002",
                    "semantic": "Stickman",
                    "blend": PEDESTRIAN_MODEL,
                    "direction": ["left", "right"],
                }
            ],
            "rider": [
                {
                    "object": "Cylinder.002",
                    "semantic": "Rider",
                    "blend": RIDER_MODEL,
                    "direction": ["left", "right"],
                }
            ],
            "car": [
                {
                    "object": "Cube.019",
                    "semantic": "Car",
                    "blend": CAR_MODEL,
                    "direction": ["left", "right", "forward", "backward"],
                }
            ],
            "other": [
                {
                    "object": "Suzanne.001",
                    "semantic": "StreetCone",
                    "blend": STREET_CONE,
                    "direction": ["left", "right"],
                },
                {
                    "object": "Suzanne.001",
                    "semantic": "Barrier",
                    "blend": BARRIER,
                    "direction": ["left", "right"],
                },
            ],
        },
    },
    (3, 2): {
        "obstacles": {
            "pedestrian": [
                {
                    "object": "Cylinder",
                    "semantic": "Stickman",
                    "blend": PEDESTRIAN_MODEL,
                    "direction": ["left", "right"],
                }
            ],
            "rider": [
                {
                    "object": "Cylinder",
                    "semantic": "Rider",
                    "blend": RIDER_MODEL,
                    "direction": ["left", "right"],
                }
            ],
            "car": [
                {
                    "object": "Cube.017",
                    "semantic": "Car",
                    "blend": CAR_MODEL,
                    "direction": ["left", "right", "forward", "backward"],
                }
            ],
            "other": [
                {
                    "object": "Suzanne.004",
                    "semantic": "StreetCone",
                    "blend": STREET_CONE,
                    "direction": ["left", "right"],
                },
                {
                    "object": "Suzanne.004",
                    "semantic": "Barrier",
                    "blend": BARRIER,
                    "direction": ["left", "right"],
                },
            ],
        }
    },
    (3, 3): {
        "camera": {"object": "Empty.005", "direction": "forward"},
        "car": {"object": "Cube.020", "direction": ["forward", "left", "right"]},
        "obstacles": {
            "pedestrian": [
                {
                    "object": "Cylinder.001",
                    "semantic": "Stickman",
                    "blend": PEDESTRIAN_MODEL,
                    "direction": ["left", "right"],
                }
            ],
            "rider": [
                {
                    "object": "Cylinder.001",
                    "semantic": "Rider",
                    "blend": RIDER_MODEL,
                    "direction": ["left", "right"],
                }
            ],
            "car": [
                {
                    "object": "Cube.020",
                    "semantic": "Car",
                    "blend": CAR_MODEL,
                    "direction": ["left", "right", "forward", "backward"],
                }
            ],
            "other": [
                {
                    "object": "Suzanne.002",
                    "semantic": "StreetCone",
                    "blend": STREET_CONE,
                    "direction": ["left", "right"],
                },
                {
                    "object": "Suzanne.002",
                    "semantic": "Barrier",
                    "blend": BARRIER,
                    "direction": ["left", "right"],
                },
            ],
        },
    },
    (4, 3): {
        "camera": {"object": "Empty.006", "direction": "forward"},
        "car": {"object": "Cube.011", "direction": "forward"},
        "obstacles": {
            "car": [
                {
                    "object": "Cube.011",
                    "semantic": "Car",
                    "blend": CAR_MODEL,
                    "direction": ["left", "right", "forward", "backward"],
                }
            ],
            "other": [
                {
                    "object": "Suzanne.006",
                    "semantic": "StreetCone",
                    "blend": STREET_CONE,
                    "direction": ["left", "right"],
                },
                {
                    "object": "Suzanne.006",
                    "semantic": "Barrier",
                    "blend": BARRIER,
                    "direction": ["left", "right"],
                },
            ],
        },
    },
    (5, 3): {
        "camera": {"object": "Empty.007", "direction": "forward"},
        "obstacles": {
            "car": [
                {
                    "object": "Cube.008",
                    "semantic": "Car",
                    "blend": CAR_MODEL,
                    "direction": ["left", "right", "forward", "backward"],
                }
            ],
            "other": [
                {
                    "object": "Suzanne.003",
                    "semantic": "StreetCone",
                    "blend": STREET_CONE,
                    "direction": ["left", "right"],
                },
                {
                    "object": "Suzanne.003",
                    "semantic": "Barrier",
                    "blend": BARRIER,
                    "direction": ["left", "right"],
                },
            ],
        },
    },
    (1, 4): {
        "semaphore": {
            "object": "Sphere.001",
            "semantic": ["TLG", "TLR", "TLY"],
            "blend": [TLG, TLR, TLY],
        },
        "sign": {
            "object": "Sphere.001",
            "semantic": [
                "Stop",
                "mph30",
            ],
            "blend": [STOP, MPH30],
        },
    },
    (2, 4): {
        "obstacles": {
            "car": [
                {
                    "object": "Cube.006",
                    "semantic": "Car",
                    "blend": CAR_MODEL,
                    "direction": ["right"],
                }
            ],
            "other": [
                {
                    "object": "Suzanne.005",
                    "semantic": "StreetCone",
                    "blend": STREET_CONE,
                    "direction": ["right"],
                },
                {
                    "object": "Suzanne.005",
                    "semantic": "Barrier",
                    "blend": BARRIER,
                    "direction": ["left"],
                },
            ],
        },
    },
    (1, 2): {
        "obstacles": {
            "car": [
                {
                    "object": "Cube.013",
                    "semantic": "Car",
                    "blend": CAR_MODEL,
                    "direction": ["backward"],
                }
            ],
        },
    },
}
