import math
import re
import time
import jieba
import numpy as np


def get_unigram_tf(tf_dic, words):
    """
    获取一元词词频
    :return:一元词词频tf_dic
    """
    for i in range(len(words)):
        tf_dic[words[i]] = tf_dic.get(words[i], 0) + 1


def get_bigram_tf(tf_dic, words):
    """
    获取二元词词频
    :return:二元词词频tf_dic
    """
    for i in range(len(words) - 1):
        tf_dic[(words[i], words[i + 1])] = tf_dic.get((words[i], words[i + 1]), 0) + 1


def get_trigram_tf(tf_dic, words):
    """
    获取三元词词频
    :return:三元词词频tf_dic
    """
    for i in range(len(words) - 2):
        tf_dic[((words[i], words[i + 1]), words[i + 2])] = tf_dic.get(((words[i], words[i + 1]), words[i + 2]), 0) + 1


def data_processing(file_path, flag):
    """
    获取文件信息并预处理
    :param file_path: 文件名对应路径
    :param flag: 选择词/字为单位，0=词，1=字
    :return data: 字符串形式的语料库
    :return words: 分词
    :return [uf_dic, bf_dic, tf_dic]: 三种模型的词频
    """
    delete_symbol = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~「」『』（）]+'
    with open(file_path, 'r', encoding='ANSI') as f:
        data = f.read()
        data = data.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
        data = re.sub(delete_symbol, '', data)
        data = data.replace('\n', '')
        data = data.replace('\u3000', '')
        f.close()

    with open('./cn_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = []
        for a in f:
            if a != '\n':
                stopwords.append(a.strip())
    for a in stopwords:
        data = data.replace(a, '')

    words = []
    if flag == 0:
        words = list(jieba.cut(data))
    elif flag == 1:
        words = [c for c in data]
    uf_dic = {}
    bf_dic = {}
    tf_dic = {}
    get_unigram_tf(uf_dic, words)
    get_bigram_tf(bf_dic, words)
    get_trigram_tf(tf_dic, words)

    return data, words, [uf_dic, bf_dic, tf_dic]


def unigram_model(dic, sum_data):
    """
    一元模型
    :param dic: 词频
    :param sum_data: 语料库字数
    :return: [模型,一元模型分词个数,一元模型不同词个数,一元模型平均词长,一元模型信息熵,运行时间]
    """
    begin = time.time()

    unigram_sum = sum([item[1] for item in dic[0].items()])  # 一元模型分词个数
    unigram_dic = len(dic[0])  # 一元模型不同词个数
    unigram_avg = round(sum_data / float(unigram_sum), 4)  # 一元模型平均词长

    entropy = 0  # 一元模型信息熵
    for item in dic[0].items():
        entropy += -(item[1] / unigram_sum) * math.log(item[1] / unigram_sum, 2)
    entropy = round(entropy, 4)

    end = time.time()
    runtime = round(end - begin, 4)

    return ['unigram', unigram_sum, unigram_dic, unigram_avg, entropy, runtime]


def bigram_model(dic):
    """
    二元模型
    :param dic: 词频
    :return: [模型,二元模型分词个数,二元模型不同词个数,二元模型平均词长,二元模型信息熵,运行时间]
    """
    begin = time.time()

    bigram_num = sum([item[1] for item in dic[1].items()])  # 二元模型分词个数
    bigram_dic = len(dic[1])  # 二元模型不同词个数
    bigram_avg = sum(len(item[0][i]) for item in dic[1].items() for i in range(len(item[0]))) / len(dic[1])
    bigram_avg = round(bigram_avg, 4)  # 二元模型平均词长

    entropy = 0  # 二元模型信息熵
    for bi_item in dic[1].items():
        jp = bi_item[1] / bigram_num
        cp = bi_item[1] / dic[0][bi_item[0][0]]
        entropy += -jp * math.log(cp, 2)
    entropy = round(entropy, 4)

    end = time.time()
    runtime = round(end - begin, 4)

    return ['bigram', bigram_num, bigram_dic, bigram_avg, entropy, runtime]


def trigram_model(dic):
    """
    三元模型
    :param dic: 词频
    :return: [模型,三元模型分词个数,三元模型不同词个数,三元模型平均词长,三元模型信息熵,运行时间]
    """
    begin = time.time()

    trigram_num = sum([item[1] for item in dic[2].items()])  # 三元模型分词个数
    trigram_dic = len(dic[2])  # 三元模型不同词个数
    trigram_avg = sum(len(item[0][i]) for item in dic[2].items() for i in range(len(item[0]))) / len(dic[2])
    trigram_avg = round(trigram_avg, 4)  # 三元模型平均词长

    entropy = 0  # 三元模型信息熵
    for tri_item in dic[2].items():
        jp = tri_item[1] / trigram_num
        cp = tri_item[1] / dic[1][tri_item[0][0]]
        entropy += -jp * math.log(cp, 2)
    entropy = round(entropy, 4)

    end = time.time()
    runtime = round(end - begin, 4)

    return ['trigram', trigram_num, trigram_dic, trigram_avg, entropy, runtime]


def information_entropy(file_path, flag):
    """
    分别计算三个模型下的信息熵
    :param file_path: 文件名对应路径
    :param flag: 选择词/字为单位，0=词，1=字
    :return: [语料库,分词,词频,一元模型结果,二元模型结果,三元模型结果,平均信息熵]
    """
    begin = time.time()
    data, words, dic = data_processing(file_path, flag)  # 数据处理
    end = time.time()
    runtime = round(end - begin, 4)

    # 以下三个值应与一元分词模型结果相同
    sum_data = len(data)  # 语料库字数
    sum_words = len(words)  # jieba分词/分字 个数
    avg_word = round(sum_data / sum_words, 4)  # jieba分词/分字 平均词长

    print('语料库字数：', sum_data)
    if flag == 0:
        print('jieba分词个数：', sum_words)
    elif flag == 1:
        print('按字分词个数：', sum_words)
    print('数据处理时间(s)：', runtime)
    print('                 模型分词个数 | 平均词长 | 信息熵 | 运行时间(s)')
    # unigram一元模型
    unigram = unigram_model(dic, sum_data)
    print('*unigram一元模型：', unigram[2], ' | ', unigram[3], ' | ', unigram[4], ' | ', unigram[5])

    # bigram二元模型
    bigram = bigram_model(dic)
    print('*bigram 二元模型：', bigram[2], ' | ', bigram[3], ' | ', bigram[4], ' | ', bigram[5])

    # trigram三元模型
    trigram = trigram_model(dic)
    print('*trigram三元模型：', trigram[2], ' | ', trigram[3], ' | ', trigram[4], ' | ', trigram[5])

    avg_entropy = np.mean([unigram[4], bigram[4], trigram[4]])  # 平均信息熵
    print('平均信息熵： %.4f' % avg_entropy)

    return [data, words, dic, unigram, bigram, trigram, avg_entropy]


if __name__ == "__main__":
    files = [['./data_novel/白马啸西风.txt'],
             ['./data_novel/碧血剑.txt'],
             ['./data_novel/飞狐外传.txt'],
             ['./data_novel/连城诀.txt'],
             ['./data_novel/鹿鼎记.txt'],
             ['./data_novel/三十三剑客图.txt'],
             ['./data_novel/射雕英雄传.txt'],
             ['./data_novel/神雕侠侣.txt'],
             ['./data_novel/书剑恩仇录.txt'],
             ['./data_novel/天龙八部.txt'],
             ['./data_novel/侠客行.txt'],
             ['./data_novel/笑傲江湖.txt'],
             ['./data_novel/雪山飞狐.txt'],
             ['./data_novel/倚天屠龙记.txt'],
             ['./data_novel/鸳鸯刀.txt'],
             ['./data_novel/越女剑.txt']]
    files_inf = ["白马啸西风", "碧血剑", "飞狐外传", "连城诀", "鹿鼎记", "三十三剑客图", "射雕英雄传", "神雕侠侣", "书剑恩仇录", "天龙八部", "侠客行", "笑傲江湖",
                 "雪山飞狐", "倚天屠龙记", "鸳鸯刀", "越女剑"]

    ci_result = {}
    print('\n\n**********以 词 为单位计算平均信息熵**********')
    for i, file in enumerate(files):
        print('----------------------------------------------------------')
        print('***当前处理文件为：《', files_inf[i], '》', '单位：词')
        ci_result[i] = information_entropy(file[0], 0)

    zi_result = {}
    print('\n\n**********以 字 为单位计算平均信息熵**********')
    for i, file in enumerate(files):
        print('----------------------------------------------------------')
        print('***当前处理文件为：《', files_inf[i], '》', '单位：字')
        zi_result[i] = information_entropy(file[0], 1)
