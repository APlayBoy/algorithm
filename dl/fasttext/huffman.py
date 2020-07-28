import numpy as np
import heapq


class PriorityQueue:
    def __init__(self):
        # 队列的内容是Huffman object
        self.queue = []

    def get_qsize(self):
        return len(self.queue)

    def insert(self, item):
        heapq.heappush(self.queue, item)

    def pop(self):
        heapq.heappop(self.queue)


class Huffman:
    def __init__(self, weight, data, embed_size):
        self.weight = weight
        self.data = data
        self.W = np.random.random((embed_size, 1))
        self.left_child = None
        self.right_child = None

    def __lt__(self, other):
        return self.weight < other.weight


def transverse(node):
    # 用来验证Huffman树搞对了没有
    if node:
        print(node.weight, " ", end="")
    if node.left_child:
        transverse(node.left_child)
    if node.right_child:
        transverse(node.right_child)


def get_min2nodes(queue):
    if len(queue) > 1:
        min1 = heapq.heappop(queue)
        min2 = heapq.heappop(queue)
        # 返回的是(weight, classObject)元组
        return min1, min2
    else:
        min1 = heapq.heappop(queue)
        return min1


def make_huffman(data, embed_size):
    '''
    :param data: data是一个列表，里面存着单个元组，即（标签，标签出现的概率）
    :return: 返回一个Huffman数的根节点，即一个Huffman类
    '''
    data_queue = PriorityQueue()
    for node in data:
        temp = Huffman(node[0], node[1], embed_size)
        data_queue.insert(temp)

    while len(data_queue.queue) > 1:
        min1, min2 = get_min2nodes(data_queue.queue)
        # 先找出最小和次小的数据，然后形成新节点，再压入队列

        sum_node = Huffman(min1.weight + min2.weight, '', embed_size)
        sum_node.left_child = min1
        sum_node.right_child = min2
        data_queue.insert(sum_node)

    return data_queue.queue[0]


