"""
Assignment 1
a. Implement BPE tokenizer
b. Implement Transformer, cross-entropy loss, AdamW optimizer, training loop
c. Train on TinyStories and OpenWebText
d. Leaderboard: minimize OpenWebText perplexity given 90 minutes on a H100
"""
import heapq


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

    def train(self, text, num_merges = 100): # English text
        # 正则匹配获得一些word chunk
        chunk_list = list(set(self.pretokenizer(text)))

        # 正向匹配 - 频率
        chunk_freq = {'<unk>': 0}
        current_idx = 0

        while current_idx < len(text):
            first_letter = text[current_idx]

            candidates = [chunk for chunk in chunk_list if chunk.startswith(first_letter)]
            candidates.sort(key=len, reverse=True)

            match = False
            for candidate in candidates:
                if text.startswith(candidate, current_idx):
                    match = True
                    current_idx += len(candidate)

                    # chunk内部划分
                    freq_name = " ".join(candidate)

                    chunk_freq[freq_name] = chunk_freq.get(freq_name, 0) + 1

                    break
                
            if not match:
                chunk_freq['<unk>'] += 1
                current_idx += 1
            
        # 生成对应的pair"_freq & 反向索引
        pair_freq = {}
        pair2chunk = {}
        for name, freq in chunk_freq.items():
            if name == '<unk>':
                continue
            for idx in range(len(name)-1):
                pair = (name[idx], name[idx+1]) # ('l','o')

                pair_freq[pair] = pair_freq.get(pair, 0) + freq
                pair2chunk[pair] = pair2chunk.get(pair, []).append(name)

        for _ in range(num_merges):      
            # 记录内部符号的全局频率 - 最大堆
            max_heap = []
            for pair, freq in pair_freq.items():
                if name == '<unk>':
                    continue
                heapq.heappush(max_heap, (-freq, pair))
            
            # 取出最高频率词，记录合并规则
            max_pair = self.max_heap[0][1]
            max_freq = - self.max_heap[0][0]
            self.merge_rule.append([max_pair])

            # 定位受影响词，然后定位受影响pair，改名
            influ_chunks = pair2chunk[max_pair]
            for chunk in influ_chunks:
                for idx in range(len(chunk)-1):
                
                    if chunk[idx] == max_pair[0] and chunk.startswith(max_pair, idx):
                        # 如果是开头
                        if idx == 0 and "".join(max_pair) == chunk:
                            past_chunk = None
                            next_chunk = None
                        if idx == 0:
                            next_chunk = (chunk[idx+1], chunk[idx+2])
                        
                        if idx + 1 == len(chunk):
                            next_chunk = None
                        else:
                            next_chunk = (chunk[idx+1], chunk[idx+2])
                
                if past_chunk:
                    new_chunk = (past_chunk[0], "".join(max_pair))
                    pair_freq[new_chunk] = pair_freq.pop(past_chunk)
                    pair2chunk[new_chunk] = pair2chunk.pop(past_chunk)
                
                if next_chunk:
                    new_chunk = (next_chunk[0], "".join(max_pair))
                    pair_freq[new_chunk] = pair_freq.pop(next_chunk)
                    pair2chunk[new_chunk] = pair2chunk.pop(next_chunk)

                if past_chunk or next_chunk:
                    _ = pair_freq.pop(max_pair)
                    _ = pair2chunk.pop(max_pair)

            
        

    
    def pretokenizer(self):



# tokenizer = BPETokenizer()
# text = 'abbbbcdelf'
# print(('on running'))
# token = tokenizer.train(text)
# print(f'token是{token}')
