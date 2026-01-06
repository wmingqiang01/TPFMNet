#!/bin/bash
###
 # @Author: 爱吃菠萝 zhangjia_liang@foxmail.com
 # @Date: 2023-10-20 23:16:39
 # @LastEditors: 爱吃菠萝 1690608011@qq.com
 # @LastEditTime: 2024-02-25 20:49:07
 # @FilePath: /root/arctic_sic_prediction/data/gen_data_text.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

DATA_DIR=.
TEXTFILE=data.txt

for year in `ls $DATA_DIR`
do
    if [ -d "${DATA_DIR}/${year}" ]; then        
        for datafile in `ls ${DATA_DIR}/${year}`
            do
                echo ${DATA_DIR}/${year}/$datafile >> $TEXTFILE
            done
    fi
done
