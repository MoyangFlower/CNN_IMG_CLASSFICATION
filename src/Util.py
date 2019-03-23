
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


class Comment(object):

    def all_path(self, dirname):
        result = []  # 所有的文件
        for maindir, subdir, file_name_list in os.walk(dirname):

            for filename in file_name_list:
                apath = os.path.join(maindir, filename)  # 合并成一个完整路径
                ext = os.path.splitext(apath)[1]  # 获取文件后缀 [0]获取的是除了文件名以外的内容

                if ext in filter:
                    result.append(apath)
        # random.shuffle(result)
        return result

    def getformat(self, key, jump_flag):

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
