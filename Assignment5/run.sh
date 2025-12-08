#!/bin/bash

# 1. 创建结果输出目录 (如果不存在)
mkdir -p results

# 2. 打印提示信息
echo "=================================================="
echo "Starting Point Cloud Diffusion Training & Generation"
echo "Course: ELEC 5630 Assignment 5"
echo "=================================================="

# 3. 运行主程序
# 注意：我们要确保使用的是 python/point_cloud_diffusion.py
# 如果你的系统 python 命令是 python3，请自行修改为 python3
python python/point_cloud_diffusion.py

# 4. 运行结束提示
echo "=================================================="
echo "Execution Finished."
echo "Results saved in 'results/' directory."
echo "Please take a screenshot of this output for submission."
echo "=================================================="