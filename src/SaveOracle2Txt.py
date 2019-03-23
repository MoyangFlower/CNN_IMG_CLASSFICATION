import os
import time
import random





if __name__ == '__main__':
    filter = [".txt"]  # 设置过滤后的文件类型 当然可以设置多个类型

    old_list = "D:\\Documents\\OracleData\\ReDownLoaded"
    new_list = "D:\\Documents\\OracleData\\transedTxt\\"

    progressbar = Index()

    count_step = 0

    # upload 0,1,2,3,4,14,24,34,44,54,64,74,84,94,104,114,124,134,144
    stop = 552

    # row_size:新txt的行是原来txt的行的20倍
    row_size = 20
    # txt_size:新txt的行数
    txt_size = 100

    start_flag = 0
    txt_name = 0

    count = 0
    newrow = ""

    newfile = open(new_list + str(txt_name) + ".txt", 'w+')
    for path in all_path(old_list):
        # print(count_step, path)
        count_step += 1
        filename = path.split("\\")[-1].replace(".txt", "")

        # family
        family = "Day"
        jump_flag = 1

        # sample rate
        if "01" in filename[-2:]:
            family = "Min"
            jump_flag = 24 * 60
        elif "00" in filename[-2:]:
            family = "Second"
            jump_flag = 24 * 60 * 60
        elif "60" in filename[-2:]:
            family = "Hour"
            jump_flag = 24

        with open(path, 'r') as f:
            for line in f.readlines():
                temp = line.replace("\n", "").split(',')
                count += 1
                # 剔除不合适的数据
                if len(temp[-1].split(" ")) > jump_flag + 1:
                    print('not one day', len(temp[-1].split(" ")))
                    continue
                if (len(temp) != 6):
                    print('not 6 item', len(temp))
                    continue
                if '00:00:00' not in temp[0].split(" ")[1]:
                    continue
                # rowkey
                rowkey = filename + temp[0].split(" ")[0].replace("-", "")

                # if value is null
                if 'NULLALL' in temp[-1]:
                    column = getformat(0, jump_flag)
                    if(start_flag == 0):
                        newrow = rowkey + ',' + family + ',' + column + ',' + 'NULLALL'
                        start_flag += 1
                    else:
                        newrow += '|' + rowkey + ',' + family + ',' + column + ',' + 'NULLALL'
                    continue

                values_list = temp[-1].split(" ")
                length_list = len(values_list)
                # for key, value in enumerate(temp[-1].split(" ")):
                for key in range(jump_flag):
                    # column

                    column = getformat(key, jump_flag)
                    # value
                    if (key < length_list):
                        value = values_list[key]
                    else:
                        value = 'NULL'

                    if(start_flag == 0):
                        newrow = rowkey + ',' + family + ',' + column + ',' + value
                        start_flag += 1
                    else:
                        newrow += '|' + rowkey + ',' + family + ',' + column + ',' + value
                # 写一行到新文件
                if(count % row_size == 0):
                    newrow += "\n"
                    newfile.write(newrow)
                    newrow = ""
                    start_flag = 0
                if(count % (row_size*txt_size) == 0):
                    newfile.close()
                    txt_name += 1
                    newfile = open(new_list + str(txt_name) + ".txt", 'w+')

        print(progressbar(count_step, stop), end='')
        time.sleep(0.01)
    newfile.close()
