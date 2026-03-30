# import json
# import os
# import re
# import pandas as pd
# from typing import List, Tuple
#
#
# def get_next_analyse_dir_num(base_dir: str = "../zzg_analyse") -> int:
#     """
#     获取zzg_analyse目录下下一个analyse_y文件夹的编号y
#     - 若目录不存在或为空，返回0
#     - 否则返回最大的y + 1
#     """
#     if not os.path.exists(base_dir):
#         os.makedirs(base_dir)
#         return 0
#
#     # 匹配analyse_数字 格式的文件夹
#     dir_pattern = re.compile(r"analyse_(\d+)")
#     max_num = -1
#
#     for dir_name in os.listdir(base_dir):
#         dir_path = os.path.join(base_dir, dir_name)
#         if os.path.isdir(dir_path):
#             match = dir_pattern.match(dir_name)
#             if match:
#                 num = int(match.group(1))
#                 if num > max_num:
#                     max_num = num
#
#     return max_num + 1 if max_num != -1 else 0
#
#
# def generate_analyse_excel(
#         exp_name: List[Tuple[int, str]],
#         traffic_intensity: List[int],
#         result_path_template: str = "../results/exp_{}/results.json"
# ):
#     """
#     生成分析用Excel文件，按指标分Sheet，自动生成输出路径
#
#     参数:
#         exp_name: 实验ID和名称的元组列表，如[(3, 'name1'), (4, 'name2')]
#         traffic_intensity: 业务强度整数列表，需与JSON中指标值列表长度一致
#         result_path_template: 结果JSON文件路径模板
#     """
#     # -------------------------- 1. 初始化参数和路径 --------------------------
#     # 获取输出目录编号
#     y = get_next_analyse_dir_num()
#     output_dir = f"../zzg_analyse/analyse_{y}"
#     output_path = os.path.join(output_dir, f"analyse_{y}.xlsx")
#
#     # 存储所有实验数据：key=(exp_id, exp_name, scene), value={metric: values}
#     all_exp_data = {}
#     # 收集所有场景和指标名称
#     all_scenes_dict = {}
#     all_scenes = set()
#     all_metrics = set()
#
#     # -------------------------- 2. 读取并验证所有实验数据 --------------------------
#     for exp_id, exp_name_str in exp_name:
#         # 拼接JSON文件路径
#         json_path = result_path_template.format(exp_id)
#         if not os.path.exists(json_path):
#             raise FileNotFoundError(f"实验{exp_id}的结果文件不存在: {json_path}")
#
#         # 读取JSON数据
#         with open(json_path, "r", encoding="utf-8") as f:
#             try:
#                 exp_data = json.load(f)
#             except json.JSONDecodeError:
#                 raise ValueError(f"实验{exp_id}的JSON文件格式错误: {json_path}")
#
#         # 移除description字段（如果存在）
#         exp_data.pop("description", None)
#         if 'traffic_intensities' in exp_data:
#             traffic_intensity = exp_data['traffic_intensities']
#         exp_data.pop("traffic_intensities", None)
#
#         # 遍历场景和指标，验证并收集数据
#         for scene, metric_dict in exp_data.items():
#             if scene not in all_scenes_dict:
#                 all_scenes_dict[scene]=1
#             else:
#                 all_scenes_dict[scene] += 1
#             for metric, values in metric_dict.items():
#                 all_metrics.add(metric)
#
#                 # 验证数值长度是否匹配业务强度列表
#                 if len(values) != len(traffic_intensity):
#                     raise ValueError(
#                         f"实验{exp_id}场景{scene}指标{metric}的值长度({len(values)}) "
#                         f"与业务强度长度({len(traffic_intensity)})不匹配"
#                     )
#
#             # 存储该实验-场景的所有指标数据
#             all_exp_data[(exp_id, exp_name_str, scene)] = metric_dict
#     for k,v in all_scenes_dict.items():
#         if v == len(exp_name) :
#             all_scenes.add(k)
#
#     # -------------------------- 3. 为每个指标生成Sheet --------------------------
#     # 创建输出目录
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     # 创建Excel写入器
#     with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
#         for metric in sorted(all_metrics):  # 按指标名排序，保证Sheet顺序固定
#             # 构建表头：业务强度 + expid1_name1_场景1 + expid1_name1_场景2 + ...
#             header = ["业务强度"]
#             for exp_id, exp_name_str in exp_name:
#                 for scene in sorted(all_scenes):  # 场景排序，保证列顺序固定
#                     header.append(f"{exp_id}_{exp_name_str}_{scene}")
#
#             # 构建数据行：第一列是业务强度，后续是对应指标值
#             data_rows = []
#             for ti_idx, ti_value in enumerate(traffic_intensity):
#                 row = [ti_value]
#                 # 填充每个(实验, 场景)对应的指标值
#                 for exp_id, exp_name_str in exp_name:
#                     for scene in sorted(all_scenes):
#                         key = (exp_id, exp_name_str, scene)
#                         row.append(all_exp_data[key][metric][ti_idx])
#                 data_rows.append(row)
#
#             # 创建DataFrame并写入Sheet
#             df = pd.DataFrame(data_rows, columns=header)
#             df.to_excel(writer, sheet_name=metric, index=False)
#
#     print(f"Excel文件已成功生成：{output_path}")
#     print(f"包含指标Sheet：{sorted(all_metrics)}")
#     print(f"包含场景：{sorted(all_scenes)}")
#
#
# # -------------------------- 示例使用 --------------------------
# if __name__ == "__main__":
#     # 1. 定义实验列表（实验ID + 实验名称）
#     exp_name_list = [
#         # (87, "baseline"),
#         # (86, "OEFM_algo_15"),
#         # (88, "OEFM"),
#         # (89, "OEFM_algo_1"),
#         # (90, "OEFM_algo_0"),
#         # 下面的实验均新增了比特率阻塞率与比特阻塞率两个指标
#         # (91, "baseline"),
#         # (92, "OEFM_algo_15"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#         # (93, "OEFM_algo_0"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#         # (94, "OEFM_algo_1"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#         # (95, "OEFM_algo_3"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#         # (96, "OEFM_algo_5"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#         (97, "n_baseline"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#         # (98, "c_baseline"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#         # (99, "c_OEFM_algo_1"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#         # (100, "c_OEFM_algo_3"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#         # (99, "c_OEFM_algo_5"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#         # (99, "c_OEFM_algo_10"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#         # (100, "c_OEFM_algo_15"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#         # (104, "u_baseline"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#         # (105, "u_OEFM_algo_1"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#         # (106, "u_OEFM_algo_3"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#         # (107, "n_OFM_algo_5"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#         (108, "n_OFM"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#         (109, "n_OEFM"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#         (110, "n_algo"),  # 新增了比特率阻塞率与比特阻塞率两个指标
#
#     ]
#
#     # 2. 定义业务强度列表（需与JSON中指标值列表长度一致，示例中是5个值）
#     traffic_intensity_list = [300, 350, 400, 450, 500]
#
#     # 3. 生成Excel分析表
#     try:
#         generate_analyse_excel(
#             exp_name=exp_name_list,
#             traffic_intensity=traffic_intensity_list
#         )
#     except Exception as e:
#         print(f"生成Excel失败：{e}")

import json
import os
import re
import pandas as pd
from typing import List, Tuple, Optional


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
        traffic_intensity: Optional[List[int]] = None,
        result_path_template: str = "../results/exp_{}/results.json",
        base_exp_id: Optional[int] = None
):
    """
    生成分析用Excel文件，按指标分Sheet，自动生成输出路径

    参数:
        exp_name: 实验ID和名称的元组列表，如[(3, 'name1'), (4, 'name2')]
        traffic_intensity: 业务强度整数列表，如果JSON中有traffic_intensities字段则优先使用JSON中的
        result_path_template: 结果JSON文件路径模板
        base_exp_id: 基准实验的ID，用于计算下降比例。如果为None，则不计算比例
    """
    # -------------------------- 1. 初始化参数和路径 --------------------------
    # 获取输出目录编号
    y = get_next_analyse_dir_num()
    output_dir = f"../zzg_analyse/analyse_{y}"
    output_path = os.path.join(output_dir, f"analyse_{y}.xlsx")

    # 存储所有实验数据：key=(exp_id, exp_name, scene), value={metric: values}
    all_exp_data = {}
    # 收集所有场景和指标名称
    all_scenes_dict = {}
    all_scenes = set()
    all_metrics = set()

    # 存储基准实验数据
    base_data = None
    base_exp_name = None
    base_traffic_intensity = None

    # 重新组织实验列表，确保基准实验在第一位（如果存在）
    organized_exp_list = []
    base_exp_info = None

    for exp_id, exp_name_str in exp_name:
        if base_exp_id is not None and exp_id == base_exp_id:
            base_exp_info = (exp_id, exp_name_str)
        else:
            organized_exp_list.append((exp_id, exp_name_str))

    # 如果存在基准实验，将其放在列表最前面
    if base_exp_info is not None:
        organized_exp_list.insert(0, base_exp_info)
        base_data = {}
        base_exp_name = base_exp_info[1]

    # 用于存储最终使用的业务强度列表
    final_traffic_intensity = None

    # -------------------------- 2. 读取并验证所有实验数据 --------------------------
    # 首先读取基准实验数据（如果存在）
    if base_exp_id is not None:
        base_json_path = result_path_template.format(base_exp_id)
        if not os.path.exists(base_json_path):
            raise FileNotFoundError(f"基准实验{base_exp_id}的结果文件不存在: {base_json_path}")

        with open(base_json_path, "r", encoding="utf-8") as f:
            try:
                base_raw_data = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"基准实验{base_exp_id}的JSON文件格式错误: {base_json_path}")

        # 获取基准实验的业务强度
        if 'traffic_intensities' in base_raw_data:
            base_traffic_intensity = base_raw_data['traffic_intensities']
            print(f"基准实验{base_exp_id}使用JSON中的traffic_intensities: {base_traffic_intensity}")

        # 移除description和traffic_intensities字段
        base_raw_data.pop("description", None)
        base_raw_data.pop("traffic_intensities", None)

        # 验证基准实验的业务强度是否匹配，并存储数据
        for scene, metric_dict in base_raw_data.items():
            base_data[scene] = {}
            for metric, values in metric_dict.items():
                # 如果基准实验有traffic_intensities，使用它进行验证
                if base_traffic_intensity is not None:
                    if len(values) != len(base_traffic_intensity):
                        raise ValueError(
                            f"基准实验{base_exp_id}场景{scene}指标{metric}的值长度({len(values)}) "
                            f"与业务强度长度({len(base_traffic_intensity)})不匹配"
                        )
                base_data[scene][metric] = values

            # 收集场景和指标
            if scene not in all_scenes_dict:
                all_scenes_dict[scene] = 1
            else:
                all_scenes_dict[scene] += 1
            for metric in metric_dict.keys():
                all_metrics.add(metric)

        # 存储基准实验数据到all_exp_data
        for scene, metric_dict in base_data.items():
            all_exp_data[(base_exp_id, base_exp_name, scene)] = metric_dict

    # 读取其他实验数据
    for exp_id, exp_name_str in organized_exp_list:
        # 跳过已经读取的基准实验
        if base_exp_id is not None and exp_id == base_exp_id:
            continue

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

        # 获取当前实验的业务强度
        current_traffic_intensity = None
        if 'traffic_intensities' in exp_data:
            current_traffic_intensity = exp_data['traffic_intensities']
            print(f"实验{exp_id}使用JSON中的traffic_intensities: {current_traffic_intensity}")

        # 验证所有实验的traffic_intensities是否一致
        if final_traffic_intensity is None:
            # 确定使用哪个业务强度
            if current_traffic_intensity is not None:
                final_traffic_intensity = current_traffic_intensity
            elif traffic_intensity is not None:
                final_traffic_intensity = traffic_intensity
            else:
                raise ValueError("既没有JSON中的traffic_intensities，也没有传入traffic_intensity参数")
        else:
            # 后续实验，验证一致性
            if current_traffic_intensity is not None:
                if current_traffic_intensity != final_traffic_intensity:
                    raise ValueError(
                        f"实验{exp_id}的traffic_intensities({current_traffic_intensity}) "
                        f"与已确定的业务强度({final_traffic_intensity})不一致"
                    )

        # 移除description和traffic_intensities字段
        exp_data.pop("description", None)
        exp_data.pop("traffic_intensities", None)

        # 遍历场景和指标，验证并收集数据
        for scene, metric_dict in exp_data.items():
            if scene not in all_scenes_dict:
                all_scenes_dict[scene] = 1
            else:
                all_scenes_dict[scene] += 1
            for metric, values in metric_dict.items():
                all_metrics.add(metric)

                # 验证数值长度是否匹配业务强度列表
                if len(values) != len(final_traffic_intensity):
                    raise ValueError(
                        f"实验{exp_id}场景{scene}指标{metric}的值长度({len(values)}) "
                        f"与业务强度长度({len(final_traffic_intensity)})不匹配"
                    )

            # 存储该实验-场景的所有指标数据
            all_exp_data[(exp_id, exp_name_str, scene)] = metric_dict

    # 如果没有从其他实验中确定业务强度，但基准实验有，则使用基准实验的
    if final_traffic_intensity is None and base_traffic_intensity is not None:
        final_traffic_intensity = base_traffic_intensity
        print(f"使用基准实验的traffic_intensities: {final_traffic_intensity}")

    # 最后验证业务强度是否已确定
    if final_traffic_intensity is None:
        raise ValueError(
            "无法确定业务强度列表，请确保至少有一个实验包含traffic_intensities字段或传入traffic_intensity参数")

    # 确定所有实验共有的场景（包括基准实验）
    total_exp_count = len(organized_exp_list)
    for scene, count in all_scenes_dict.items():
        if count == total_exp_count:
            all_scenes.add(scene)

    # 如果存在基准实验，确保只使用共有场景
    valid_scenes = all_scenes

    # -------------------------- 3. 为每个指标生成Sheet --------------------------
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建Excel写入器
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for metric in sorted(all_metrics):  # 按指标名排序，保证Sheet顺序固定
            # 构建表头：业务强度 + 所有实验的数据列（按organized_exp_list顺序）
            header = ["业务强度"]

            # 所有实验的数据列（包括基准实验）
            all_data_columns = []
            for exp_id, exp_name_str in organized_exp_list:
                for scene in sorted(valid_scenes):
                    col_name = f"{exp_id}_{exp_name_str}_{scene}"
                    header.append(col_name)
                    all_data_columns.append(col_name)

            # 比例列（如果存在基准实验）- 只对其他实验计算比例
            ratio_columns = []
            if base_data is not None and len(organized_exp_list) > 1:
                # 跳过第一个（基准实验），对其他实验添加比例列
                for exp_id, exp_name_str in organized_exp_list[1:]:  # 从第二个实验开始
                    for scene in sorted(valid_scenes):
                        ratio_col_name = f"ratio_{exp_id}_{exp_name_str}_{scene}"
                        header.append(ratio_col_name)
                        ratio_columns.append(ratio_col_name)

            # 构建数据行：第一列是业务强度，后续是对应指标值和比例值
            data_rows = []
            num_scenes = len(valid_scenes)

            for ti_idx, ti_value in enumerate(final_traffic_intensity):
                row = [ti_value]

                # 填充所有实验的指标值
                exp_values = []  # 存储当前行的所有实验值，用于后续计算比例
                for exp_id, exp_name_str in organized_exp_list:
                    for scene in sorted(valid_scenes):
                        key = (exp_id, exp_name_str, scene)
                        if key in all_exp_data and metric in all_exp_data[key]:
                            value = all_exp_data[key][metric][ti_idx]
                        else:
                            value = None  # 数据缺失时填充None
                        row.append(value)
                        exp_values.append((exp_id, exp_name_str, scene, value))

                # 计算比例列（如果有基准实验）
                if base_data is not None and len(organized_exp_list) > 1:
                    # 获取基准实验的值（前num_scenes个）
                    base_values = exp_values[:num_scenes]

                    # 对其他实验计算比例
                    for i in range(num_scenes, len(exp_values)):
                        exp_id, exp_name_str, scene, value = exp_values[i]
                        # 找到对应场景的基准值
                        base_value = None
                        for b_exp_id, b_exp_name, b_scene, b_value in base_values:
                            if b_scene == scene:
                                base_value = b_value
                                break

                        if base_value is not None and base_value != 0:
                            if value is not None:
                                ratio = (value - base_value) / base_value
                            else:
                                ratio = None
                        elif base_value == 0:
                            ratio = float('inf') if value and value > 0 else 0
                        else:
                            ratio = None
                        row.append(ratio)

                data_rows.append(row)

            # 创建DataFrame并写入Sheet
            df = pd.DataFrame(data_rows, columns=header)
            df.to_excel(writer, sheet_name=metric, index=False)

            # 可选：设置比例列的格式为百分比（需要openpyxl）
            if base_data is not None and ratio_columns:
                worksheet = writer.sheets[metric]
                # 找到比例列的索引位置
                for col_name in ratio_columns:
                    col_idx = header.index(col_name) + 1  # Excel列索引从1开始
                    # 处理列字母（支持超过26列的情况）
                    col_letter = ''
                    temp = col_idx
                    while temp > 0:
                        temp -= 1
                        col_letter = chr(65 + temp % 26) + col_letter
                        temp //= 26

                    for row_idx in range(2, len(data_rows) + 2):  # 从第2行开始（跳过表头）
                        cell = worksheet[f"{col_letter}{row_idx}"]
                        if cell.value is not None and cell.value != float('inf'):
                            cell.number_format = '0.00%'
                        elif cell.value == float('inf'):
                            cell.value = 'INF'

    print(f"Excel文件已成功生成：{output_path}")
    print(f"使用的业务强度列表：{final_traffic_intensity}")
    print(f"包含指标Sheet：{sorted(all_metrics)}")
    print(f"包含场景：{sorted(valid_scenes)}")
    print(f"实验顺序：{[f'{exp_id}_{exp_name}' for exp_id, exp_name in organized_exp_list]}")
    if base_data is not None:
        print(f"基准实验：{base_exp_id}_{base_exp_name}（位于第一列）")
        print(f"已添加下降比例列（格式：ratio_expid_expname_scene）")


# -------------------------- 示例使用 --------------------------
if __name__ == "__main__":
    # 1. 定义实验列表（实验ID + 实验名称）
    exp_name_list = [
        # (87, "baseline"),
        # (86, "OEFM_algo_15"),
        # (88, "OEFM"),
        # (89, "OEFM_algo_1"),
        # (90, "OEFM_algo_0"),
        # 下面的实验均新增了比特率阻塞率与比特阻塞率两个指标
        # (91, "baseline"),
        # (92, "OEFM_algo_15"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (93, "OEFM_algo_0"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (94, "OEFM_algo_1"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (95, "OEFM_algo_3"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (96, "OEFM_algo_5"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (97, "n_baseline"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (98, "c_baseline"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (99, "c_OEFM_algo_1"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (100, "c_OEFM_algo_3"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (99, "c_OEFM_algo_5"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (99, "c_OEFM_algo_10"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (100, "c_OEFM_algo_15"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (104, "u_baseline"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (105, "u_OEFM_algo_1"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (106, "u_OEFM_algo_3"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (107, "n_OFM_algo_5"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (108, "n_OFM"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (109, "n_OEFM"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (110, "n_algo"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (118, "n_baseline"),  # 更正了label算法
        # 30000
        # (119, "n_baseline"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (120, "n_OFM"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (121, "n_OEFM"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (122, "n_algo_0"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (123, "n_algo_3"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # (124, "n_OFM_algo_3"),  # 新增了比特率阻塞率与比特阻塞率两个指标
        # 增加了EON层碎片格式：EONz
        (126, "n_baseline"),
        (127, "n_EONz"),
        (128, "n_algo"),
        (129, "n_EONz_algo_3"),
        (130, "n_ABP_baseline"),

        # 无优势
        # (131, "c_ABP_base"),
        # (134, "c_EONz"),
        # (133, "c_EONz_algo"),




    ]
    base_exp_id = 126
    # 2. 定义业务强度列表（如果JSON中有traffic_intensities字段，将优先使用JSON中的）
    traffic_intensity_list = [300, 350, 400, 450, 500]  # 作为备用值

    # 3. 生成Excel分析表，指定基准实验ID为97
    try:
        generate_analyse_excel(
            exp_name=exp_name_list,
            traffic_intensity=traffic_intensity_list,  # 作为备用，如果JSON中有则会被覆盖
            base_exp_id=base_exp_id  # 指定基准实验
        )
    except Exception as e:
        print(f"生成Excel失败：{e}")