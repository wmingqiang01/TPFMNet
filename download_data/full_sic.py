"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2024-02-25 18:20:25
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2024-02-25 21:01:30
FilePath: /root/arctic_sic_prediction/data/full_sic.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""

# 通过调用 write_netcdf 函数并将上一步的输出文件名data.txt作为参数传递，读取并存储处理过的数据，以方便下次读取数据
from utils import write_netcdf

start_time = 198801
end_time = 202408
write_netcdf("data.txt", start_time, end_time, "full_sic.nc")
