import os
from typing import List
from collections import Counter
import matplotlib.pyplot as plt
from utils import read_text
import numpy as np

datasets = ['eurlex-4k', 'wiki10-31k', 'amazoncat-13k', 'wiki-500k']
biases = [-1.5, -0.5, 0.5, 1.5]  # 柱子偏移, 四个数据集为[-1.5, -0.5, 0.5, 1.5], 三个数据集为[-1, 0, 1]
divide = 200
width = 0.2  # 柱子宽度，四个数据集为0.2，三个数据集为0.25
word_counts = []
ratios = []
word_range = []
totals = []
for dataset in datasets:
    datadir = os.path.join('dataset', dataset, 'X.tst.txt')
    texts = read_text(datadir)
    word_count = []
    ratio = []
    total = 0
    for text in texts:
        space_num = Counter(text)[' ']
        word_num = space_num + 1
        if word_num >= 1000:
            # word_num = 1000
            continue
        id = word_num // divide
        total = total + 1
        while len(word_count) <= id:
            word_count.append(0)
        while len(word_range) <= id:
            word_range.append(str(len(word_range) * divide) + '-' + str((len(word_range) + 1) * divide))
        word_count[id] = word_count[id] + 1
    word_counts.append(word_count)
    totals.append(total)
# word_range[-1] = ">1000"
for id, word_count in enumerate(word_counts):
    while len(word_count) < len(word_range):
        word_count.append(0)
    ratio = []
    for c in word_count:
        ratio.append(c / totals[id])
    ratios.append(ratio)

x = np.arange(len(word_range))
fig = plt.figure(1)
for id, _ in enumerate(datasets):
    plt.bar(x + width * biases[id], ratios[id], width, label=datasets[id])
plt.ylabel('ratio')
plt.xlabel('text length')
plt.title('text length distribution')
plt.xticks(x, labels=word_range)
plt.legend()
fig.savefig(os.path.join('word_count_figure'))
plt.show()
