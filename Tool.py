from enum import Enum
import os
import re

# Constants and parameters
class TrafficClass(Enum):
    GE10 = 10
    GE100 = 100
    GE400 = 400

class EventType(Enum):
    ARRIVAL = 1
    DEPARTURE = 2

class Event:
    def __init__(self, event_type, time, demand=None):
        self.event_type = event_type
        self.time = time
        self.demand = demand

    def __lt__(self, other):
        return self.time < other.time


def get_next_exp_number(output_path):
    """获取下一个实验编号"""
    if not os.path.exists(output_path):
        return 0

    # 查找所有已存在的 exp_数字 目录
    existing_dirs = [d for d in os.listdir(output_path)
                     if os.path.isdir(os.path.join(output_path, d)) and re.match(r'exp_\d+', d)]

    if not existing_dirs:
        return 0

    # 提取数字并找到最大值
    numbers = []
    for dir_name in existing_dirs:
        try:
            num = int(dir_name.split('_')[1])
            numbers.append(num)
        except (IndexError, ValueError):
            continue

    if numbers:
        return max(numbers) + 1
    else:
        return 0