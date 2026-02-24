"""
Assignment 1
a. Implement BPE tokenizer
b. Implement Transformer, cross-entropy loss, AdamW optimizer, training loop
c. Train on TinyStories and OpenWebText
d. Leaderboard: minimize OpenWebText perplexity given 90 minutes on a H100
"""
import heapq
from collections import defaultdict


class BPETokenizer():

    def __init__(self):
        self.vocab_list = []
        self.ori_char_freq = {}
        self.decode_table = {}
        self.encode_table = {}
    
    def train(self, ori_text, num_merges = 100):

        # ver3 刷新token initial
        token = list(ori_text)
        
        refresh = 1
        while refresh:
            num_merges -= 1
            
            freq_dict = {}

            # 统计pair频率
            for idx in range(len(token) - 1):
                pair = token[idx] + token[idx+1]
                freq_dict[pair] = freq_dict.get(pair, 0) + 1
            
            if not freq_dict:
                break
            
            # 找到最大的pair
            max_pair = max(freq_dict, key=freq_dict.get)
            if freq_dict[max_pair] == 1:
                break

            # 按照最大的pair刷新token / 正向匹配
            current_idx = 0
            refresh = None
            new_token = []
            while current_idx < len(token) - 1:
                # if ori_text[current_idx] == max_pair[0] and ori_text.startswith(max_pair, current_idx): # 直接对token而不是原始文本
                if current_idx <= len(token) - 1 and token[current_idx] + token[current_idx + 1] == max_pair: # pair是附近+1的token组合，所以可以直接在token上操作而不是在原始文本上操作
                    new_token.append(max_pair)
                    current_idx += 2
                    refresh = 1
                else:
                    new_token += [token[current_idx]]
                    current_idx += 1
            
            token = new_token
            
            if num_merges == 0:
                break


        token = sorted(list(set(token)))      
        # 最新token 映射
        for idx, char in enumerate(token):
            self.encode_table[char] = idx
            self.decode_table[idx] = char
        
        return token

# rewrite
class BPETokenizer2(): 

    def __init__(self):
        self.vocab_list = []
        self.merge_rule = []

    def train(self, text, chunk_dict,num_merges = 100): # English text
        # chunk _dict = {chunk1: freq, chunk2: freq,...}
        # init part
        pair_freq = defaultdict(int) # pair = tuple (a,b)
        pair_chunk = defaultdict(set)
        chunk_pair = defaultdict(list)
        heap = []
        for chunk, freq in chunk_dict.items():
            chunk_list = list(chunk)
            for idx in range(len(chunk_list) - 1):
                # 分解chunk得到pair_freq
                pair = (chunk_list[idx], chunk_list[idx + 1])
                pair_freq[pair] += freq
                # 维护pair to chunk
                pair_chunk[pair].add(chunk) # 这里collection.defaultdict 的用法/ 以及add和append<返回None>    PPPPPPPPPPPPoint1
                chunk_pair[chunk].add(pair)
       
        
        # 维护最大堆 pair_freq -> heap
        for pair, freq in pair_freq.items():
           heapq.heappush(heap, (-freq, pair))  # 维护堆                        PPPPPPPPPPPPoint2
        
        # loop part
        while num_merges > 0 or heap[0][0] != -1:
            max_freq, max_pair = heap[0]
            max_freq = - max_freq
            self.merge_rule.append(max_pair)

            # 定位pair 影响的pair
            chunks = pair_chunk[max_pair] # [chunk1, chunk2, ...]
            for chunk in chunks:          
                

        
        
               

           

            
        
    def pretokenizer(self) -> dict: 
        chunk_dict ={}

        return chunk_dict



# tokenizer = BPETokenizer()
# text = 'abbbbcdelf'
# print(('on running'))
# token = tokenizer.train(text)
# print(f'token是{token}')
