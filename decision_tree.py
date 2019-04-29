import codecs
import os
import re
from time import time

import jieba
from sklearn import tree
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

category = ['culture', 'education', 'entertainment', 'event', 'fashion', 'health', 'occultism', 'sport', 'technology',
            'travel']


# 从data文件夹加载数据
# 标签和新闻的格式

def load_data(path, to_path, per_class_docs=1000):
    corpus = []
    with open(to_path, 'w') as f:
        for files in os.listdir(path):
            curr_path = os.path.join(path, files)
            print(curr_path)
            if os.path.isdir(curr_path):
                count = 0
                docs = []
                for file in os.listdir(curr_path):
                    count = count + 1
                    if count > per_class_docs:
                        break
                    file_path = os.path.join(curr_path, file)
                    with codecs.open(file_path, 'r', encoding='utf-8') as fp:
                        docs.append(
                            files + '\t' + ' '.join(jieba.cut(re.sub('[ \n\r\t]+', '', fp.read()))))
                        # f.write('__label__' + files + ' ' + ' '.join(jieba.cut(re.sub('[ \n\r\t]+', '', fp.read()))))
            corpus.append(docs)
        with codecs.open(to_path, 'w', encoding='utf-8') as f:
            for docs in corpus:
                for doc in docs:
                    f.write(doc + '\n')
        return corpus


# 保存标签和数据
def split_data_with_label(corpus):
    """将数据划分新闻和样本标签"""
    tag = []
    input_y = []
    input_x = []
    if os.path.isfile(corpus):
        with codecs.open(corpus, 'r', encoding='utf-8') as f:
            for line in f:
                assert len(line.split('\t')) == 2
                label, content = line.split('\t')
                input_y.append(label)
                input_x.append(content)
    return [input_x, input_y]


# 使用tf-idf特征抽取
def feature_extractor(input_x, case='tfidf', max_df=1.0, min_df=0.0):
    """特征抽取"""
    return TfidfVectorizer(token_pattern='\w', ngram_range=(1, 2), max_df=max_df, min_df=min_df).fit_transform(input_x)


# 划分训练集和测试集
def split_data_to_train_test(filename, indices=0.2, random_state=10, shuffle=True):
    input_x, input_y = split_data_with_label(filename)
    input_x = feature_extractor(input_x, 'tfidf')
    x_train, x_dev, y_train, y_dev = train_test_split(input_x, input_y, test_size=indices, random_state=random_state)
    print("Vocabulary Size: {:d}".format(input_x.shape[1]))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, x_dev, y_train, y_dev


def decision(train_x, train_y, test_x, test_y):
    clf = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=4)
    clf.fit(train_x, train_y)
    predicted = clf.predict(test_x)
    print(metrics.classification_report(test_y, predicted))
    print("accuracy:%0.5f" % (metrics.accuracy_score(test_y, predicted)))


path = 'data'
to_path = 'news_data'
corpus = load_data(path, to_path, per_class_docs=1000)
print(len(corpus), len(corpus[0]))

x_train, x_dev, y_train, y_dev = split_data_to_train_test('news_data')
t0 = time()

print('\t\t使用 max_df,min_df=(1.0,0.0) 进行特征选择的决策树文本分类\t\t')
decision(x_train, y_train, x_dev, y_dev)
print('time uesed: %0.4fs' % (time() - t0))
