"""
Assignment 1
a. Implement BPE tokenizer
b. Implement Transformer, cross-entropy loss, AdamW optimizer, training loop
c. Train on TinyStories and OpenWebText
d. Leaderboard: minimize OpenWebText perplexity given 90 minutes on a H100
"""

class BPETokenizer():

    def __init__(self):
        self.vocab_list = []
        self.ori_char_freq = {}
        self.decode_table = {}
        self.encode_table = {}
    
    def train(self, ori_text, num_merges = 100):
        # token_list = sorted(list(set(ori_text)))

        # for char in ori_text:
        #     self.ori_char_freq[char] = self.ori_char_freq.get(char, 0) + 1 # z

        # # ver1.循环的对象不清楚，为什么是ori_text，自己没理解就写上去了
        # # # init condition
        # # for char in ori_text:
        # #     self.char_freq[char] = self.char_freq.get(char, 0) + 1 # freq_table
        


        # # # loop part

        # # for idx, char in enumerate(ori_text[:-1]):
        # #     pair = ori_text[idx] + ori_text[idx+1]
        # #     self.pair_freq[pair] = self.pair_freq.get(pair, 0) + 1  # pair freq table
        # #     if self.pair_freq[pair] >= 2:
        # #         token_list += pair  # Add into chars and decline in char_freq
        # #         self.char_freq[pair] = self.char_freq.get(pair, 0) + 1
                
        # #         self.char_freq[char] -= 1
        # #         self.char_freq[ori_text[idx+1]] -= 1

        # # ver2. 提前创建pair表格，确认是否存在2，直到pair value =1   //// ver2 这条路走不通
        
        # # 原始token_freq
        # init_token_freq = {}
        # for idx, char in enumerate(text):
        #     init_token_freq[char] = init_token_freq.get(char, 0) + 1

        # # 两两组合，原文匹配
        # token_freq = init_token_freq
        # merge = 1
        # while merge:
        #     num_merges -= 1
        #     merge = None

        #     token_list = list(token_freq.keys())

        #     # pair_freq 
        #     pair_freq = {}
        #     for idx in range(len(token_list) - 1):
        #         pair_tuple = (token_list[idx], token_list[idx + 1])  # 组合不够完全

        #         pair = pair_tuple[0] + pair_tuple[1]

        #         # 匹配
        #         current_idx = 0
                
        #         while current_idx < len(text) - 1:
        #             if text[current_idx] == pair[0] and text.startswith(pair, current_idx):
        #                 pair_freq[pair_tuple] = pair_freq.get(pair_tuple, 0) + 1
        #                 current_idx += len(pair)
        #             else:
        #                 current_idx += 1
                
        #     # 选出max_pair
        #     max_pair_tuple = max(pair_freq, key=pair_freq.get)
        #     freq_time = pair_freq[max_pair_tuple]
        #     if freq_time == 1:
        #         break
        #     pair_str = max_pair_tuple[0] + max_pair_tuple[1]

        #     # 更新token_freq
        #     token_freq[pair_str] = freq_time
        #     for char in max_pair_tuple:
        #         token_freq[char] -= freq_time
            
        #     merge = 1

        #     if num_merges == 0:
        #         break

        # return token_freq.keys()
        
        # ver2 存不存在什么问题，还有没有优化的版本？ 
        # ver2 不符合业界的BPE逻辑，业界是刷新token来实现的，而不是在频率表上加减实现的 -- 但是能修改这部分来能同样达到作用吗？




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
                    

    # encode / decode都不需要在token里面操作，只需要等token结果出来之后直接进行分配就可以了
    # def encode(self, text):
        
    #     # 首字母索引
    #     first_letter_index = {}
    #     for token in self.vocab_list:
    #         first_letter = token[0]
    #         if first_letter not in first_letter_index:
    #             first_letter_index[first_letter] = []

    #         first_letter_index[first_letter].append(token)

    #     # 最大排序
    #     for char in first_letter_index:
    #         first_letter_index[char].sort(key=len, reverse=True)

    #     # 最大正向匹配
    #     candidates = []

    #     current_idx = 0
    #     encode_result = []

    #     while current_idx < len(text):
    #         letter = text[current_idx]
    #         matched = None

    #         candidates = first_letter_index.get(letter, [])

    #         for candidate in candidates:
    #             if text.startswith(candidate, current_idx):
    #                 matched = candidate
    #                 break
            
    #         if matched:
    #             encode = self.encode_table[matched]
    #             encode_result.append(encode)
    #             current_idx += len(matched)
    #         else:
    #             encode_result.append(0) # 0 代表 <UNK> 未知
    #             i += 1
            
    #     return encode_result


    # def decode(self, nums):

    #     decode_result = []

    #     for digit in nums:
    #         token = self.decode_table[digit]
    #         decode_result.append(token)
        
    #     text = ''.join(decode_result)

    #     return text


tokenizer = BPETokenizer()
text = 'abbbbcdelf'
print(('on running'))
token = tokenizer.train(text)
print(f'token是{token}')