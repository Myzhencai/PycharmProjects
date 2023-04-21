#基于Keras框架实现加入Attention与BiRNN的机器翻译模型
# https://zhuanlan.zhihu.com/p/37290775
import warnings
warnings.filterwarnings("ignore")

from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, Reshape
from keras.layers import RepeatVector, Dense, Activation, Lambda, Embedding
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import keras
import numpy as np
import random
import tqdm
import matplotlib.pyplot as plt

# English source data
with open("data/small_vocab_en", "r", encoding="utf-8") as f:
    source_text = f.read()

# French target data
with open("data/small_vocab_fr", "r", encoding="utf-8") as f:
    target_text = f.read()

view_sentence_range = (0, 10)

# Separate the source language text by spaces, to see how many distinct words are contained in it
print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))

# 统计英文语料数据
print("-"*5 + "English Text" + "-"*5)
sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))
print('Max number of words in a sentence: {}'.format(np.max(word_counts)))

# 统计法语语料数据
print()
print("-"*5 + "French Text" + "-"*5)
sentences = target_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))
print('Max number of words in a sentence: {}'.format(np.max(word_counts)))

# 打印语料的前10个句子
print()
print('English sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('French sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))

# 构造英文词典
source_vocab = list(set(source_text.lower().split()))
# 构造法语词典
target_vocab = list(set(target_text.lower().split()))

print("The size of English vocab is : {}".format(len(source_vocab)))
print("The size of French vocab is : {}".format(len(target_vocab)))

# 增加特殊编码
SOURCE_CODES = ['<PAD>', '<UNK>']
TARGET_CODES = ['<PAD>', '<EOS>', '<UNK>', '<GO>']

# 构造英文语料的映射表
source_vocab_to_int = {word: idx for idx, word in enumerate(SOURCE_CODES + source_vocab)}
source_int_to_vocab = {idx: word for idx, word in enumerate(SOURCE_CODES + source_vocab)}

# 构造法语语料的映射表
target_vocab_to_int = {word: idx for idx, word in enumerate(TARGET_CODES + target_vocab)}
target_int_to_vocab = {idx: word for idx, word in enumerate(TARGET_CODES + target_vocab)}

print("The size of English Map is : {}".format(len(source_vocab_to_int)))
print("The size of French Map is : {}".format(len(target_vocab_to_int)))


def text_to_int(sentence, map_dict, max_length=20, is_target=False):
    """
    Encoding the text into integers.

    @param sentence: 完整的句子，str类型
    @param map_dict: 单词到数字编码的映射
    @param max_length: 最大句子长度
    @param is_target: 当前传入的句子是否是目标语句。
                      对于目标语句，我们要在末尾添加"<EOS>"
    """

    text_to_idx = []
    # 特殊单词的数字编码
    unk_idx = map_dict.get("<UNK>")
    pad_idx = map_dict.get("<PAD>")
    eos_idx = map_dict.get("<EOS>")

    # 如果不是目标语句（即源语句）
    if not is_target:
        for word in sentence.split():
            text_to_idx.append(map_dict.get(word, unk_idx))

    # 目标语句要对结尾添加"<EOS>"
    else:
        for word in sentence.split():
            text_to_idx.append(map_dict.get(word, unk_idx))
        text_to_idx.append(eos_idx)

    # 超长句子进行截断
    if len(text_to_idx) > max_length:
        return text_to_idx[:max_length]
    # 不足长度的句子进行"<PAD>"
    else:
        text_to_idx = text_to_idx + [pad_idx] * (max_length - len(text_to_idx))
        return text_to_idx

# 对英文语料进行编码，其中设置英文句子最大长度为20
Tx = 20
source_text_to_int = []

for sentence in tqdm.tqdm(source_text.split("\n")):
    source_text_to_int.append(text_to_int(sentence, source_vocab_to_int, Tx, is_target=False))

random_index = 77

# 对法语语料进行编码，其中设置法语句子最大长度为25
Ty = 25
target_text_to_int = []

for sentence in tqdm.tqdm(target_text.split("\n")):
    target_text_to_int.append(text_to_int(sentence, target_vocab_to_int, Ty, is_target=True))

print("-"*5 + "English example" + "-"*5)
print(source_text.split("\n")[random_index])
print(source_text_to_int[random_index])

print()
print("-"*5 + "French example" + "-"*5)
print(target_text.split("\n")[random_index])
print(target_text_to_int[random_index])

from keras.utils import to_categorical
X = np.array(source_text_to_int)
Y = np.array(target_text_to_int)
# 对X和Y做One Hot Encoding
Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(source_vocab_to_int)), X)))
Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(target_vocab_to_int)), Y)))

# 自定义softmax函数
def softmax(x, axis=1):
    """
    Softmax activation function.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')

# 定义全局网络层对象
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor_tanh = Dense(32, activation = "tanh")
densor_relu = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes = 1)


def one_step_attention(a, s_prev):
    """
    Attention机制的实现，返回加权后的Context Vector

    @param a: BiRNN的隐层状态
    @param s_prev: Decoder端LSTM的上一轮隐层输出

    Returns:
    context: 加权后的Context Vector
    """

    # 将s_prev复制Tx次
    s_prev = repeator(s_prev)
    # 拼接BiRNN隐层状态与s_prev
    concat = concatenator([a, s_prev])
    # 计算energies
    e = densor_tanh(concat)
    energies = densor_relu(e)
    # 计算weights
    alphas = activator(energies)
    # 加权得到Context Vector
    context = dotor([alphas, a])

    return context