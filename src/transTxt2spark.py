import os
import time
import random

# 读出目录下的文件并打乱


def all_path(dirname, end_filter):
    result = []  # 所有的文件
    for main_dir, subdir, file_name_list in os.walk(dirname):

        for file_name in file_name_list:
            a_path = os.path.join(main_dir, file_name)  # 合并成一个完整路径
            ext = os.path.splitext(a_path)[1]  # 获取文件后缀 [0]获取的是除了文件名以外的内容

            if ext in end_filter:
                result.append(a_path)
    random.shuffle(result)
    return result


def getformat(key, jump_flag):

    hour = min = ""
    if (jump_flag == 24*60):

        if (key // 60) < 10:
            hour = '0' + str(key // 60)
        else:
            hour = str(key // 60)
        if ((key % 60) < 10):
            min = '0' + str(key % 60)
        else:
            min = str(key % 60)

    mtime = str(hour) + str(min)
    # print(mtime)
    return mtime


class Index(object):
    def __init__(self, number=50, decimal=2):
        """
        :param decimal: 你保留的保留小数位
        :param number: # 号的 个数
        """
        self.decimal = decimal
        self.number = number
        self.a = 100 / number  # 在百分比 为几时增加一个 # 号

    def __call__(self, now, total):
        # 1. 获取当前的百分比数
        percentage = self.percentage_number(now, total)

        # 2. 根据 现在百分比计算
        well_num = int(percentage / self.a)
        # print("well_num: ", well_num, percentage)

        # 3. 打印字符进度条
        progress_bar_num = self.progress_bar(well_num)

        # 4. 完成的进度条
        result = "\r%s %s" % (progress_bar_num, percentage)
        return result

    def percentage_number(self, now, total):
        """
        计算百分比
        :param now:  现在的数
        :param total:  总数
        :return: 百分
        """
        return round(now / total * 100, self.decimal)

    def progress_bar(self, num):
        """
        显示进度条位置
        :param num:  拼接的  “#” 号的
        :return: 返回的结果当前的进度条
        """
        # 1. "#" 号个数
        well_num = "#" * num

        # 2. 空格的个数
        space_num = " " * (self.number - num)

        return '[%s%s]' % (well_num, space_num)


if __name__ == '__main__':

    filter_end = [".txt"]  # 设置过滤后的文件类型 当然可以设置多个类型

    # QZ_411_DYU_01BAKQZ_411_DYU_01BAK,QZ_911_DYS_01BAK,QZ_913_DYS_01BAK

    old_list = "D:\\Documents\\OracleData\\QZ_913_DYS_01BAK"
    new_list = "D:\\Documents\\OracleData\\transedTxt913\\"

    progressbar = Index()
    # 已读取文件数，为进度条
    count_step = 0
    stop = 552
    # upload 0,1,2,3,4,14,24,34,44,54,64,74,84,94,104,114,124,134,144

    # row_size:新txt的行是原来txt的行的200倍
    row_size = 5
    # txt_size:新txt的行数
    txt_size = 8000
    # 是否是每行的开始
    start_flag = 0
    # 新文件名
    txt_name = 0
    # 遍历old文件的行数
    row_count = 0
    new_row = ""

    new_file = open(new_list + str(txt_name) + ".txt", 'w+')

    for path in all_path(old_list, filter_end):
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
                row_count += 1

                # 剔除不合适的数据
                if len(temp[-1].split(" ")) > jump_flag + 1:
                    print('not one day', filename, temp[0], len(temp[-1].split(" ")))
                    continue
                if len(temp) != 6:
                    print('not 6 item', len(temp))
                    continue
                if '00:00:00' not in temp[0].split(" ")[1]:
                    continue
                # row_key
                row_key = filename + temp[0].split(" ")[0].replace("-", "")

                # if value is null
                if 'NULLALL' in temp[-1]:
                    column = getformat(0, jump_flag)
                    if start_flag == 0:
                        new_row = row_key + ',' + family + ',' + column + '&' + 'NULLALL'
                        start_flag += 1
                    else:
                        new_row += '|' + row_key + ',' + family + ',' + column + '&' + 'NULLALL'
                    continue

                values_list = temp[-1].split(" ")
                length_list = len(values_list)

                # 处理第一列，添加row_key和family信息

                column = getformat(0, jump_flag)
                if start_flag == 0:
                    new_row = row_key + ',' + family + ',' + column + '&' + values_list[0]
                    start_flag += 1
                else:
                    new_row += '|' + row_key + ',' + family + ',' + column + '&' + values_list[0]

                # 处理第二及以后的列，只包含column和value

                for key in range(1, jump_flag):
                    # column

                    column = getformat(key, jump_flag)
                    # value
                    if key < length_list:
                        value = values_list[key]
                    else:
                        value = 'NULL'
                    new_row += ' ' + column + '&' + value

                # 每row_size行就写一行到新文件
                if row_count % row_size == 0:
                    new_row += "\n"
                    new_file.write(new_row)
                    new_file.close()
                    new_file = open(new_list + str(txt_name) + ".txt", 'a+')
                    new_row = ""
                    start_flag = 0

                # 每txt_size行就保存一个文件
                if row_count % (row_size*txt_size) == 0:
                    new_file.close()
                    txt_name += 1
                    new_file = open(new_list + str(txt_name) + ".txt", 'w+')

        print(progressbar(count_step, stop), end='')
        time.sleep(0.01)
    new_file.close()
