# KG-XMC
a standard XMC solution repository by KG method

## Data prosessing

## dataset

X.txt:文章raw text \
Y.npz:文章的label index --> 稀疏矩阵spacy\
output-items.txt：index和实体标签对应文件\
tfidf-attnxml/X.npz:  文章的tfidf向量

how to access dataset:
In the top dir(terminal) \
runing 
`data.sh wiki10-31k`
shellscript can download dataset files and unzip them automatically.
## Model overivew

## pipeline
after runing `data.sh` \
runing  `run.sh` \
run.sh script can control pipeline and hyperparameters.

## Git
Github link is `https://github.com/CountyRipper/KG-XMC.git`


## 2023-09-25
This project was first from [2021.9 enrollment] Dai Xiangting
DElab [2022.9 enrollment] Li Jiahao start editing \

kg_type=" [prefix-] [two-] bart[l] [-kpdrop_{a/r/na/nr}] [-shuffle] "

## code structure
models: {keyphrase_generation_model, combine_model, rank_model}
utils: {utils, evaluation_on_generated_key_words, text_length_distribution}

## contact
any problem in this project from DElab student is welcome.
my wechat: 18381796772
or my e-mail: 2173219824@qq.com
2024-03-14



