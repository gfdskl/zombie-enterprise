import os
import glob
import pickle
import copy
import numpy as np
import pandas as pd

from preprocess import preProcess


dtypes = [
    {'ID': "Int64", '注册时间': "Int64", '注册资本': np.float64, '行业': object, '区域': object,
        '企业类型': object, '控制人类型': object, '控制人持股比例': np.float64, 'flag': "Int64"},
    "Int64",
    {'ID': "Int64", 'year': "Int64", '债权融资额度': np.float64, '债权融资成本': np.float64, '股权融资额度': np.float64, '股权融资成本': np.float64,
        '内部融资和贸易融资额度': np.float64, '内部融资和贸易融资成本': np.float64, '项目融资和政策融资额度': np.float64, '项目融资和政策融资成本': np.float64},
    {'ID': "Int64", 'year': np.float64, '从业人数': np.float64, '资产总额': np.float64, '负债总额': np.float64, '营业总收入': np.float64, '主营业收入': np.float64, '利润总额': np.float64, '净利润': np.float64, '纳税总额': np.float64, '所有者权益合计': np.float64}]

model_path = "/Users/sameal/Documents/PROJECT/zombie_enterprise/code/model/XGBClassifier_web.pickle"


def get_file_path(dir_path):
    base_path = glob.glob(os.path.join(dir_path, "*base*.csv"))[0]
    knowledge_path = glob.glob(os.path.join(dir_path, "*knowledge*.csv"))[0]
    money_path = glob.glob(os.path.join(dir_path, "*money*.csv"))[0]
    year_path = glob.glob(os.path.join(dir_path, "*year*.csv"))[0]
    return (base_path, knowledge_path, money_path, year_path)


def load_data(dir_path):
    file_path = get_file_path(dir_path)
    data = []
    for i in range(4):
        table = pd.read_csv(file_path[i], encoding='gbk', dtype=dtypes[i])
        table.set_index("ID", inplace=True)
        data.append(table)
    return data


def year_fea(yb_df):
    f1 = lambda row: row["净利润"] / row["营业总收入"] if row["营业总收入"] else None
    f2 = lambda row: row["净利润"] / row["资产总额"] if row["资产总额"] else None
    f3 = lambda row: row["负债总额"] / row["资产总额"] if row["资产总额"] else None
    f4 = lambda row: row["净利润"] / row["所有者权益合计"] if row["所有者权益合计"] else None
    yb_df["销售净利率"] = yb_df.apply(f1, axis=1)
    yb_df["资产净利率"] = yb_df.apply(f2, axis=1)
    yb_df["资产负债率"] = yb_df.apply(f3, axis=1)
    yb_df["资金收益率"] = yb_df.apply(f4, axis=1)


def portrait(data):
    base, knowledge, money, year = copy.deepcopy(data)
    base["注册资本"] = base["注册资本"] / 10

    creat_label = knowledge.sum(axis=1)
    creat_label.name = "创新能力"
    
    year_fea(year)
    year_mean = year.groupby('ID').mean().drop(columns="year")
    yb_mean = pd.concat([base, year_mean], axis=1)

    def scale_classify(row):
        if row["行业"] == "交通运输业":
            x, y = row["从业人数"], row["营业总收入"]
            return ((x >= 20) + (y >= 100) + (x >= 300) + (y >= 3000) + (x >= 1000) + (y >= 30000)) // 2
        if row["行业"] == "工业":
            x, y = row["从业人数"], row["营业总收入"]
            return ((x >= 20) + (y >= 300) + (x >= 300) + (y >= 2000) + (x >= 1000) + (y >= 40000)) // 2
        if row["行业"] == "零售业":
            x, y = row["从业人数"], row["营业总收入"]
            return ((x >= 10) + (y >= 100) + (x >= 50) + (y >= 500) + (x >= 300) + (y >= 20000)) // 2
        if row["行业"] in ["服务业", "商业服务业", "社区服务"]:
            x, y = row["从业人数"], row["资产总额"]
            return ((x >= 10) + (y >= 100) + (x >= 100) + (y >= 8000) + (x >= 300) + (y >= 120000)) // 2
        x = row["从业人数"]
        return (x >= 10) + (x >= 100) + (x >= 300)

    scale_label = yb_mean.apply(scale_classify, axis=1)
    scale_label.name = "企业规模"
    scales = ["微型企业", "小型企业", "中型企业", "大型企业"]
    yb_mean["企业规模"] = scale_label.apply(lambda row: scales[row])

    sel_columns = ["净利润", "纳税总额", "销售净利率", "资产净利率", "资产负债率", "资金收益率"]
    cut_bins = {
        "净利润": [-np.inf, -1e4, 0,  1e4, np.inf],
        "纳税总额": [-np.inf, 0, 1e4, 1e5, np.inf],
        "销售净利率": [-np.inf, -0.1, 0, 0.2, np.inf],
        "资产净利率": [-np.inf, -0.2, 0, 0.4, np.inf],
        "资产负债率": [-np.inf, 0.4, 0.6, 2, np.inf],
        "资金收益率": [-np.inf, -1, 0, 1, np.inf]
    }
    sel_df = year_mean[sel_columns]
    sel_df = sel_df.apply(lambda col: pd.cut(col, cut_bins[col.name], labels=[0, 1, 2, 3])).astype(int)
    label_df = pd.concat([base[["注册时间", "注册资本", "行业", "区域", "企业类型"]], scale_label, creat_label, sel_df], axis=1)
    label_columns = sel_columns + ["企业规模", "创新能力"]
    label_df[label_columns] = label_df[label_columns].fillna(4).astype(int)
    return yb_mean, label_df


def analyse(dir_path):
    data = load_data(dir_path)
    mean_df, label_df = portrait(data)
    mean_df.to_csv(os.path.join(dir_path, "chart_data.csv"), header=True, index=True, index_label="ID")
    label_df.to_csv(os.path.join(dir_path, "portrait.csv"), header=True, index=True, index_label="ID")


def predict(dir_path):
    base, _, _, year = load_data(dir_path)
    year = year.drop(columns="year").groupby("ID").mean()
    year_fea(year)
    union_table = pd.concat([base["注册资本"], year], axis=1, join_axes=[base.index])

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    pred_flag = model.predict(union_table)
    res = pd.Series(pred_flag, index=union_table.index, name="pred_flag")
    save_path = os.path.join(dir_path, "result.csv")
    res.to_csv(save_path, header=True, index=True, index_label="ID")


def search(dir_path, search_id):
    portrait_path = os.path.join(dir_path, "portrait.csv")
    flag_path = os.path.join(dir_path, "result.csv")
    label_df = pd.read_csv(portrait_path)
    label_df.set_index("ID", inplace=True, drop=False)
    flag_df = pd.read_csv(flag_path)
    flag_df.set_index("ID", inplace=True)
    search_id = int(search_id)
    if not search_id in label_df.index:
        return None
    res_label = label_df.loc[search_id]
    res_label["flag"] = int(flag_df.loc[search_id].values[0] > 0)
    return res_label.to_json()


def chart(dir_path, search_id, byclass):
    chart_path = os.path.join(dir_path, "chart_data.csv")
    chart_df = pd.read_csv(chart_path)
    chart_df.set_index("ID", inplace=True)
    search_id = int(search_id)
    if not search_id in chart_df.index:
        return None
    columns = ["从业人数", "资产总额", "负债总额", "营业总收入", "主营业务收入", "利润总额", "净利润", "纳税总额", "所有者权益合计", "销售净利率", "资产净利率", "资产负债率", "资金收益率"]
    row = chart_df.loc[search_id]
    if byclass == "all":
        tmp = chart_df[columns]
    else:
        tmp = chart_df[chart_df[byclass] == row[byclass]]
        tmp = tmp[columns]
    mean = tmp.mean()
    std = tmp.std()
    row = row[columns]
    res = (row - mean) / std
    return res.to_json(orient="values")


if __name__ == "__main__":
    search("/Users/sameal/Documents/PROJECT/zombie_enterprise/web/uploads/ct79s9wf", 28)