import numpy as np
from .huffman import make_huffman


def generate_huffman_code(tree, label2code, now_code):
    if tree.left_child:
        now_code.append('1')
        generate_huffman_code(tree.left_child, label2code, now_code)
    if tree.right_child:
        now_code.append('0')
        generate_huffman_code(tree.right_child, label2code, now_code)
    if (not tree.left_child) and (not tree.right_child):
        label2code[tree.data] = "".join(now_code)
    if now_code:
        now_code.pop()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class FastText:
    def __init__(self, label_data, label2code, voca_size, embed_size):
        # 得到的是层次softmax的Huffman树的根节点
        self.embed_size = embed_size
        self.embedding = 2 * np.random.random((voca_size, embed_size)) - 1
        self.huffmanTree = make_huffman(label_data, embed_size)
        self.x = np.random.normal((embed_size, 1))
        self.label2code = label2code
        self.loss = 0
        self.raw_input = []

    def forward(self, raw_input):
        # forward计算的结果是这个输入向量对应的标签,也就是一个string
        self.raw_input = raw_input
        # 输入是raw_input也就是one-hot编码的单词列表, 其实也就是用十进制数表示的单次[1, 2 ,3]这样
        input = [self.embedding[i].T for i in raw_input]
        input = np.array(input)
        input = input[:, :, np.newaxis]
        temp = np.zeros((self.embed_size, 1))
        for single in input:
            temp += single
        temp /= len(input)
        self.x = temp

        now_node = self.huffmanTree
        p_total = 1
        while now_node.left_child or now_node.right_child:
            # print(now_node.W)
            p = np.dot(self.x.T, now_node.W)[0][0]
            p = sigmoid(p)
            if p > 0.5:
                p_total *= p
                now_node = now_node.right_child
            else:
                p_total *= (1 - p)
                now_node = now_node.left_child
        # print(p_total)
        return now_node.data

    def backward(self, target_label, lr):
        e = np.zeros((self.embed_size, 1))
        code = self.label2code[target_label]
        now_node = self.huffmanTree
        while now_node.left_child or now_node.right_child:
            g = (1 - int(code[0]) - sigmoid(np.dot(self.x.T, now_node.weight)[0][0])) * lr
            e += g * now_node.W
            dLoss_W = g * self.x
            now_node.W += lr * dLoss_W
            if code[0] == '0':
                now_node = now_node.right_child
                code = code[1:]
            else:
                now_node = now_node.left_child
                code = code[1:]
        # print(e.squeeze().shape)
        # update word embedding
        for i in self.raw_input:
            self.embedding[i] += e.T.squeeze()

    def cal_loss(self, target_label):
        loss = 0
        target_label_code = self.label2code[target_label]
        length = len(target_label_code)
        now_node = self.huffmanTree
        while now_node.left_child or now_node.right_child:
            p = np.dot(self.x.T, now_node.W)[0][0]
            p = sigmoid(p)
            if target_label_code[0] == '0':
                loss += (1 - p)
                target_label_code = target_label_code[1:]
                now_node = now_node.right_child
            else:
                loss += p
                target_label_code = target_label_code[1:]
                now_node = now_node.left_child
        return loss / length

