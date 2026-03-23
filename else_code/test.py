import os
import numpy as np
# def compare_files_simple(dir1, dir2, filename="request_results.txt"):
#     """
#     简化版本，只输出第一个不同字符的行号
#     """
#     file1_path = os.path.join(dir1, filename)
#     file2_path = os.path.join(dir2, filename)
#
#     if not os.path.exists(file1_path) or not os.path.exists(file2_path):
#         print("文件不存在")
#         return
#
#     try:
#         with open(file1_path, 'r', encoding='utf-8') as f1, \
#                 open(file2_path, 'r', encoding='utf-8') as f2:
#
#             line_num = 1
#
#             while True:
#                 line1 = f1.readline()
#                 line2 = f2.readline()
#
#                 if not line1 and not line2:
#                     print("文件相同")
#                     return
#
#                 if not line1 or not line2 or line1 != line2:
#                     print(f"第一个不同出现在第 {line_num} 行")
#                     return
#
#                 line_num += 1
#
#     except Exception as e:
#         print(f"错误: {e}")
#
# # import pickle
# # import CAG
# #
# #
# # # 从文件加载
# # with open('d.pkl', 'rb') as f:
# #     loaded_instance = pickle.load(f)
# #     print(loaded_instance.attribute)  # 输出: example
# #
# #
# # 使用示例
# dir1 = "./results/exp_59"  # 替换为实际目录
# dir2 = "./results/exp_60"  # 替换为实际目录
# compare_files_simple(dir1, dir2)

for i in range(10,410,10):
    print(f"GE{i} = {i}")