import json
import os
import re
import pandas as pd
from typing import List, Tuple


def get_next_analyse_dir_num(base_dir: str = "../zzg_analyse") -> int:
    """
    获取zzg_analyse目录下下一个analyse_y文件夹的编号y
    - 若目录不存在或为空，返回0
    - 否则返回最大的y + 1
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return 0

    # 匹配analyse_数字 格式的文件夹
    dir_pattern = re.compile(r"analyse_(\d+)")
    max_num = -1

    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.isdir(dir_path):
            match = dir_pattern.match(dir_name)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num

    return max_num + 1 if max_num != -1 else 0


def generate_analyse_excel(
        exp_name: List[Tuple[int, str]],
        traffic_intensity: List[int],
        result_path_template: str = "../results/exp_{}/results.json"
):
    """
    生成分析用Excel文件，按指标分Sheet，自动生成输出路径

    参数:
        exp_name: 实验ID和名称的元组列表，如[(3, 'name1'), (4, 'name2')]
        traffic_intensity: 业务强度整数列表，需与JSON中指标值列表长度一致
        result_path_template: 结果JSON文件路径模板
    """
    # -------------------------- 1. 初始化参数和路径 --------------------------
    # 获取输出目录编号
    y = get_next_analyse_dir_num()
    output_dir = f"../zzg_analyse/analyse_{y}"
    output_path = os.path.join(output_dir, f"analyse_{y}.xlsx")

    # 存储所有实验数据：key=(exp_id, exp_name, scene), value={metric: values}
    all_exp_data = {}
    # 收集所有场景和指标名称
    all_scenes = set()
    all_metrics = set()

    # -------------------------- 2. 读取并验证所有实验数据 --------------------------
    for exp_id, exp_name_str in exp_name:
        # 拼接JSON文件路径
        json_path = result_path_template.format(exp_id)
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"实验{exp_id}的结果文件不存在: {json_path}")

        # 读取JSON数据
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                exp_data = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"实验{exp_id}的JSON文件格式错误: {json_path}")

        # 移除description字段（如果存在）
        exp_data.pop("description", None)

        # 遍历场景和指标，验证并收集数据
        for scene, metric_dict in exp_data.items():
            all_scenes.add(scene)
            for metric, values in metric_dict.items():
                all_metrics.add(metric)

                # 验证数值长度是否匹配业务强度列表
                if len(values) != len(traffic_intensity):
                    raise ValueError(
                        f"实验{exp_id}场景{scene}指标{metric}的值长度({len(values)}) "
                        f"与业务强度长度({len(traffic_intensity)})不匹配"
                    )

            # 存储该实验-场景的所有指标数据
            all_exp_data[(exp_id, exp_name_str, scene)] = metric_dict

    # -------------------------- 3. 为每个指标生成Sheet --------------------------
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建Excel写入器
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for metric in sorted(all_metrics):  # 按指标名排序，保证Sheet顺序固定
            # 构建表头：业务强度 + expid1_name1_场景1 + expid1_name1_场景2 + ...
            header = ["业务强度"]
            for exp_id, exp_name_str in exp_name:
                for scene in sorted(all_scenes):  # 场景排序，保证列顺序固定
                    header.append(f"{exp_id}_{exp_name_str}_{scene}")

            # 构建数据行：第一列是业务强度，后续是对应指标值
            data_rows = []
            for ti_idx, ti_value in enumerate(traffic_intensity):
                row = [ti_value]
                # 填充每个(实验, 场景)对应的指标值
                for exp_id, exp_name_str in exp_name:
                    for scene in sorted(all_scenes):
                        key = (exp_id, exp_name_str, scene)
                        row.append(all_exp_data[key][metric][ti_idx])
                data_rows.append(row)

            # 创建DataFrame并写入Sheet
            df = pd.DataFrame(data_rows, columns=header)
            df.to_excel(writer, sheet_name=metric, index=False)

    print(f"Excel文件已成功生成：{output_path}")
    print(f"包含指标Sheet：{sorted(all_metrics)}")
    print(f"包含场景：{sorted(all_scenes)}")


# -------------------------- 示例使用 --------------------------
if __name__ == "__main__":
    # 1. 定义实验列表（实验ID + 实验名称）
    exp_name_list = [
        # (87, "baseline"),
        (86, "OEFM_algo_15"),
        # (88, "OEFM"),
        (89, "OEFM_algo_1"),
        # (90, "OEFM_algo_0"),
        # 下面的实验均新增了比特率阻塞率与比特阻塞率两个指标
        # (91, "baseline"),
        # (92, "OEFM_algo_15"),  # 新增了比特率阻塞率与比特阻塞率两个指标

    ]

    # 2. 定义业务强度列表（需与JSON中指标值列表长度一致，示例中是5个值）
    traffic_intensity_list = [300, 350, 400, 450, 500]

    # 3. 生成Excel分析表
    try:
        generate_analyse_excel(
            exp_name=exp_name_list,
            traffic_intensity=traffic_intensity_list
        )
    except Exception as e:
        print(f"生成Excel失败：{e}")