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
    
    def train(self, ori_text, num_merges = 100): # 缺少了pre_tokenization -> 反向索引 / 最大堆维护 /// 重新思考逻辑流程

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


# tokenizer = BPETokenizer()
# text = 'abbbbcdelf'
# print(('on running'))
# token = tokenizer.train(text)
# print(f'token是{token}')
