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
    
    # 以我目前的能力，无法写出只修改merge两侧的频率来实现更高效的bpe的内容
    # def tokenizer(self, chunks_pair):
    #     # 根据chunk_pair来确认每个pair的freq, chunk_pair = {chunk1: freq1,}
    #     word_content = defaultdict(list)
    #     word_freq = defaultdict(int)
    #     pair_freq = defaultdict(int)  # pair_freq = {pair1:freq, pair2: freq...}
    #     pair_chunk = defaultdict(set) # pair_chunk = {pair1: (chunk_list1, chunk_list2)}
    #     for word_id, (chunk, freq) in enumerate(chunks_pair.items()):
    #         # 确保 chunk 是不可变的 tuple
    #         chunk_list = list(chunk) 
            
    #         # 登记病历本 (登记实体和频率)
    #         word_content[word_id] = chunk_list
    #         word_freq[word_id] = freq
            
    #         # 统计初始的 pairs，并建立倒排索引
    #         for i in range(len(chunk_list) - 1):
    #             pair = (chunk_list[i], chunk_list[i + 1])
    #             pair_freq[pair] += freq
    #             # 这里存的是 word_id，再也不是实体 tuple 了！
    #             pair_chunk[pair].add(word_id)


    #     heap = []
    #     for pair, freq in pair_freq.items():
    #         heapq.heappush(heap, (-freq, pair))
        
    #     max_pair = heap[0][1]
    #     max_freq = - heap[0][0]
    #     self.merge_rules.append(max_pair)

    #     # check influenced chunk
    #     ids = pair_chunk[max_pair] # ids
    #     # 开始计算
    #     for id in ids:
    #         chunk = word_content[id]
    #         # 更新频率 同时记录一下prev，merge，next的dict
    #         idx = 0
    #         prev_freq = defaultdict(int)
    #         merge_freq = 0
    #         next_freq = defaultdict(int)
    #         while idx < len(chunk): # chunk ['','','']
    #             if (chunk[idx], chunk(idx+1)) == max_pair:
    #                 merge_freq += 1
    #                 # 存在这么一个被merge的部分
    #                 # 计算是否存在prev和next
    #                 prev = 1 if idx > 0 else None
    #                 next = 1 if idx + 2 <len(chunk) else None

    #                 if prev:
    #                     old_last_pair = (chunk[idx-1], chunk[idx])
    #                     new_old_pair = (chunk[idx-1], "".join(max_pair))

    #                     pair_freq[old_last_pair] -= word_freq[id]
    #                     pair_freq[new_old_pair] += word_freq[id]

    #                     prev_freq[old_last_pair] = 0

    #                 if next:
    #                     old_next_pair = (chunk[idx+1], chunk[idx+2])
    #                     new_next_pair = ("".join(max_pair), chunk[idx+2])

    #                     pair_freq[old_next_pair] -= word_freq[id]
    #                     pair_freq[new_next_pair] += word_freq[id]

    #                     next_freq[old_last_pair] = 0
                    
    #                 # 减去频率
    #                 pair_freq[max_pair] -= word_freq[id]
    #                 i += 2
    #             else:
    #                 i += 1
            
    #         # 统计 chunk内prev和next出现的次数
    #         for idx in range(len(chunk) - 1):
    #             if (chunk[idx], chunk[idx+1]) in prev_freq:
    #                 prev_freq[(chunk[idx], chunk[idx+1])] += 1
    #             if (chunk[idx], chunk[idx+1]) in next_freq:
    #                 next_freq[(chunk[idx], chunk[idx+1])] += 1
    #         # 走完一个chunk，开始计算pair_chunk的变化了
    #         # pair_chunk通过prev_freq,merge, next_freq来计算是否要在pair_chunk里面排除某些chunk
    #         # 对于前面的pre
    #         if prev_freq:
    #             for prev, freq in prev_freq.items():
    #                 # 在当前chunk里面，我出现次数并不全部首merge rules影响
    #                 if freq > merge_freq:
    #                     # 不排除
    #                     continue
    #                 # 否则，我们总是成对出现，那么prev -> chunk就要少了这一个了
                    

 


# tokenizer = BPETokenizer()
# text = 'abbbbcdelf'
# print(('on running'))
# token = tokenizer.train(text)
# print(f'token是{token}')
