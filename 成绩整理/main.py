import pandas as pd
import numpy as np
import os


def assign_grades_by_rank(raw_grades):
    """
    根据学生的原始分数分配等级。

    参数:
    raw_grades -- 学生的原始分数列表

    返回:
    assign_grades -- 分配的分数列表
    """

    # 计算每个学生的排名
    ranks = raw_grades.rank(method='min', ascending=False)

    # 处理最大排名的情况，即负数分数的情况，将其设置为 0
    ranks = np.array(ranks.mask(ranks == ranks.max(), 0))

    # 计算非零排名的数量
    num = np.count_nonzero(ranks)

    # 转换排名为百分比
    ranks = ranks / num

    # 定义等级比例和对应的得分区间
    grades_proportion = np.array([0.0, 0.15, 0.5, 0.85, 0.98, 1.0])
    grades_grade_interval = np.array([100.0, 85.0, 70.0, 55.0, 40.0, 30.0])

    # 初始化分配的分数列表
    assign_grades = []

    # 遍历每个学生的排名，使用线性插值方法为其分配分数
    for i in range(ranks.shape[0]):
        if ranks[i] > 0.0:
            assign_grade = np.interp(ranks[i], grades_proportion, grades_grade_interval)
        else:
            assign_grade = 0.0
        assign_grades.append(assign_grade)

    # 返回分配的分数列表
    return np.array(assign_grades)


def assign_grades_by_mean(raw_grades):
    """
    根据学生的原始分数分配分数。

    参数:
    raw_grades -- 学生的原始分数列表

    返回:
    assign_grades -- 分配的分数列表
    """

    # 将负数分数替换为 0
    raw_grades_ = np.array(raw_grades.mask(raw_grades < -0.5, 0))

    # 计算非零分数的数量
    num = np.count_nonzero(raw_grades_)

    # 计算平均分数
    mean = np.sum(raw_grades_) / num

    # 初始化分配的分数列表
    assign_grades = []

    # 遍历每个学生的原始分数，减去平均分数后分配分数
    for i in range(raw_grades.shape[0]):
        if raw_grades[i] > -0.5:
            assign_grade = raw_grades[i] - mean
        else:
            assign_grade = 0.0
        assign_grades.append(assign_grade)

    # 返回分配的分数列表
    return np.array(assign_grades)


if __name__ == '__main__':
    """
    原始分.csv 存储信息的格式如下(均为原始分)
    学号/班级/姓名/语文/数学/英语/物理/化学/生物/历史/地理/政治
    ps：除姓名外，其他数据均为阿拉伯数字。若某考生未考该科目，例如 A只考了语数英 物地政则其原始分数应登记为
    **/**/**/**/-1/-1/**/**。即未考科目按-1作为标识。
    """
    os.makedirs("按均值赋分", exist_ok=True)
    os.makedirs("按排名赋分", exist_ok=True)
    raw = pd.read_csv('原始分.csv', encoding='gbk')

    # 根据排名赋分
    assign_rank = raw.copy(deep=True)
    assign_rank['化学'] = assign_grades_by_rank(assign_rank['化学'])
    assign_rank['生物'] = assign_grades_by_rank(assign_rank['生物'])
    assign_rank['地理'] = assign_grades_by_rank(assign_rank['地理'])
    assign_rank['政治'] = assign_grades_by_rank(assign_rank['政治'])
    total = assign_rank[["语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "政治"]].sum(axis=1)
    assign_rank = assign_rank.assign(总分=total)
    assign_rank.to_csv('按排名赋分/总表.csv', index=False)

    classes_list = [pd.DataFrame(columns=raw.columns) for _ in range(30)]
    for _, row in assign_rank.iterrows():
        index = row['班级']
        classes_list[index] = pd.concat([classes_list[index], pd.DataFrame([row])], ignore_index=True)
    for i in range(30):
        if classes_list[i].shape[0] > 0:
            classes_list[i].to_csv(f'按排名赋分/{i}班.csv', index=False)
            print(f"按排名赋分，{i}班平均分为：{classes_list[i]['总分'].mean():.2f}")

    print("*********************************")

    # 根据均值赋分
    assign_mean = raw.copy(deep=True)
    assign_mean['化学'] = assign_grades_by_mean(assign_mean['化学'])
    assign_mean['生物'] = assign_grades_by_mean(assign_mean['生物'])
    assign_mean['地理'] = assign_grades_by_mean(assign_mean['地理'])
    assign_mean['政治'] = assign_grades_by_mean(assign_mean['政治'])
    total = assign_rank[["语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "政治"]].sum(axis=1)
    assign_mean = assign_mean.assign(总分=total)
    assign_mean.to_csv('按均值赋分/总表.csv', index=False)

    classes_list = [pd.DataFrame(columns=raw.columns) for _ in range(30)]
    for _, row in assign_mean.iterrows():
        index = row['班级']
        classes_list[index] = pd.concat([classes_list[index], pd.DataFrame([row])], ignore_index=True)
    for i in range(30):
        if classes_list[i].shape[0] > 0:
            classes_list[i].to_csv(f'按均值赋分/{i}班.csv', index=False)
            print(f"按均值赋分，{i}班平均分为：{classes_list[i]['总分'].mean():.2f}")

