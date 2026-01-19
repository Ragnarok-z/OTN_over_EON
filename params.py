# Transponder operational modes (from Table V in the paper)
TRANSPONDER_MODES_original = [
    {"capacity": 200, "bitrate": 200, "baudrate": 95, "fs_required": 19, "max_spans": 125},
    {"capacity": 300, "bitrate": 300, "baudrate": 95, "fs_required": 19, "max_spans": 88},
    {"capacity": 400, "bitrate": 400, "baudrate": 95, "fs_required": 19, "max_spans": 54},
    {"capacity": 500, "bitrate": 500, "baudrate": 95, "fs_required": 19, "max_spans": 35},
    {"capacity": 600, "bitrate": 600, "baudrate": 95, "fs_required": 19, "max_spans": 18},
    {"capacity": 700, "bitrate": 700, "baudrate": 95, "fs_required": 19, "max_spans": 9},
    {"capacity": 800, "bitrate": 800, "baudrate": 95, "fs_required": 19, "max_spans": 4},
    {"capacity": 100, "bitrate": 100, "baudrate": 56, "fs_required": 12, "max_spans": 130},
    {"capacity": 200, "bitrate": 200, "baudrate": 56, "fs_required": 12, "max_spans": 61},
    {"capacity": 300, "bitrate": 300, "baudrate": 56, "fs_required": 12, "max_spans": 34},
    {"capacity": 400, "bitrate": 400, "baudrate": 56, "fs_required": 12, "max_spans": 10},
    {"capacity": 100, "bitrate": 100, "baudrate": 35, "fs_required": 8, "max_spans": 75},
    {"capacity": 200, "bitrate": 200, "baudrate": 35, "fs_required": 8, "max_spans": 16}
]


length_of_span = 80 # length of a span is 80km
# selects the transponder operational mode that minimizes spectral usage while maximizes data rate
TRANSPONDER_MODES = [
    {"capacity": 200, "bitrate": 200, "baudrate": 35, "fs_required": 8, "max_spans": 16},
    {"capacity": 100, "bitrate": 100, "baudrate": 35, "fs_required": 8, "max_spans": 75},
    {"capacity": 400, "bitrate": 400, "baudrate": 56, "fs_required": 12, "max_spans": 10},
    {"capacity": 300, "bitrate": 300, "baudrate": 56, "fs_required": 12, "max_spans": 34},
    {"capacity": 200, "bitrate": 200, "baudrate": 56, "fs_required": 12, "max_spans": 61},
    {"capacity": 100, "bitrate": 100, "baudrate": 56, "fs_required": 12, "max_spans": 130},
    {"capacity": 800, "bitrate": 800, "baudrate": 95, "fs_required": 19, "max_spans": 4},
    {"capacity": 700, "bitrate": 700, "baudrate": 95, "fs_required": 19, "max_spans": 9},
    {"capacity": 600, "bitrate": 600, "baudrate": 95, "fs_required": 19, "max_spans": 18},
    {"capacity": 500, "bitrate": 500, "baudrate": 95, "fs_required": 19, "max_spans": 35},
    {"capacity": 400, "bitrate": 400, "baudrate": 95, "fs_required": 19, "max_spans": 54},
    {"capacity": 300, "bitrate": 300, "baudrate": 95, "fs_required": 19, "max_spans": 88},
    {"capacity": 200, "bitrate": 200, "baudrate": 95, "fs_required": 19, "max_spans": 125},
]

# Traffic engineering policy coefficients (from Table IV in the paper)
POLICY_COEFFICIENTS = {
    "MinEn": {"c0": 100000, "c0_new": 10000, "c_new": 1000, "cf": 10, "c_prime": 0, "cu": 0, "c_old": 1000},
    "MaxMux": {"c0": 0, "c0_new": 1, "c_new": 100, "cf": 0, "c_prime": 0, "cu": 0, "c_old": 0},
    "MaxSE": {"c0": 0, "c0_new": 0, "c_new": 1, "cf": 0, "c_prime": 0, "cu": 1, "c_old": 1},
    "MinPB": {"c0": 0.000001, "c0_new": 0, "c_new": 1000, "cf": 0, "c_prime": 0.001, "cu": 0, "c_old": 1000},
    "OneFrag": {"c0": 0.000001, "c0_new": 0, "c_new": 1000, "cf": 10, "c_prime": 0.001, "cu": 0, "c_old": 1000}
}