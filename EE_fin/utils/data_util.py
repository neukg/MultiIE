import re
import unicodedata

from transformers import BertTokenizer, BasicTokenizer

def search(tok2char_span, start_ind, end_ind):
    '''
    由于start_ind已经给定，因此只需在tok2char_span中找到以start_ind为开头的char_span,即可找到起始tok的位置
    '''
    for tok_ind, char_span in enumerate(tok2char_span):
        if start_ind == char_span[0]:
            for end_char_span in tok2char_span[tok_ind:]:
                if end_ind == end_char_span[-1]:
                    return tok_ind
    
    return -1

def argument_in_sent(sent, argument_list, trigger):
        """argument_in_sent"""
        trigger_start = sent.find(trigger)
        if trigger_start < 0:
            return trigger_start, []
        new_arguments = []
        for argument in argument_list:
            word = argument["argument"]
            role_type = argument["role"]
            if role_type == "环节":
                continue
            start = sent.find(word)
            if start < 0:
                continue
            new_arguments.append({
                "role": role_type,
                "argument": word,
                "argument_start_index": start
            })
        return trigger_start, new_arguments


def cut_sentence(text, max_len, sliding_window):
    split_texts = []
    split_starts = []
    sent_len = len(text)
    start = 0
    end = start + max_len
    while start < sent_len:
        cut_text = text[start: end]
        split_texts.append(cut_text)
        split_starts.append(start)
        if end > sent_len:
            break
        start += sliding_window
        end = start + max_len
    return split_texts, split_starts


#####################################################################################################################
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    if not text:
        return []
    tokens = text.split()

    return tokens 


class NewBertTokenizer(BertTokenizer):

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 do_basic_tokenize=True,
                 never_split=None,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 tokenize_chinese_chars=True,
                 **kwargs
    ):

        super().__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            **kwargs,
        )

        if do_basic_tokenize:
            self.basic_tokenizer = NewBasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
            )
        self.wordpiece_tokenizer = NewWordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)
        self.do_lower_case = do_lower_case

    def _tokenize(self, text, return_offset=False):
        split_tokens = []
        token_mappings = []

        tokens, token_starts = self.basic_tokenizer.tokenize(text, return_offset=return_offset, never_split=self.all_special_tokens)

        for token, token_start in zip(tokens, token_starts):

            # If the token is part of the never_split set
            if token in self.basic_tokenizer.never_split:
                split_tokens.append(token)
                token_mappings.append([token_start, token_start + len(token)])
            else:

                if return_offset:
                    split_token, token_mapping = self.wordpiece_tokenizer.tokenize(token, start=token_start)
                    token_mappings += token_mapping
                else:
                    split_token = self.wordpiece_tokenizer.tokenize(token)
                split_tokens += split_token
                
        if return_offset:
            return split_tokens, token_mappings
        else:
            return split_tokens
    
    def get_offset_mappings(self, text):
        return self._tokenize(text, return_offset=True)[1]

    def stem(self, token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token   

    def _is_control(self, ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    def _is_special(self, ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')
    
class NewBasicTokenizer(BasicTokenizer):

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, tokenize_english_word=True, tokenize_number=True):
        super().__init__(
            do_lower_case=do_lower_case,
            never_split=never_split,
            tokenize_chinese_chars=tokenize_chinese_chars
        )
        self.tokenize_english_word = tokenize_english_word
        self.tokenize_number = tokenize_number
    
    def _nomalize_text(self, text):
        if self.do_lower_case:
            text = text.lower()

        normalized_text = ''
        for ch in text:
            if self.do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            normalized_text += ch
        return normalized_text
    
    
    def _is_control(self, ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    def tokenize(self, text, return_offset=False, never_split=None):
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split

        if return_offset:
            normalized_text = self._nomalize_text(text)
            token_offsets = []

        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
            
        if self.tokenize_english_word:
            text = self._tokenize_english_word(text)
        
        if self.tokenize_number:
            text = self._tokenize_number(text)
        
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))

        if return_offset:
            offset = 0
            for token in output_tokens:
                start = normalized_text[offset:].index(token) + offset
                end = start + len(token)
                token_offsets.append(start)
                offset = end
            return output_tokens, token_offsets

        return output_tokens, [0]*len(output_tokens)

    def _tokenize_english_word(self, text):
        output = []
        '''
        # 英文不按字符划分的情况
        for ind, char in enumerate(text[0:-1]):
            behind_char = text[ind + 1]
            if behind_char.isalpha() and not char.isalpha():
                output.append(char)
                output.append(" ")
                
            elif not behind_char.isalpha() and char.isalpha():
                output.append(char)
                output.append(" ")
            
            else:
                output.append(char)
        output.append(text[-1])
        '''
        for char in text:
            if char.isalpha():
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)        
        return "".join(output)

    def _tokenize_number(self, text):
        output = []
        for char in text:
            if self._is_number(char):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_number(self, char):
        return False if unicodedata.numeric(char, -1) == -1 else True

class NewWordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word


    def tokenize(self, token, start=None):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        
        return_offset = True if start != None else False

        if return_offset:
            token_offsets = []

        offset = start
        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
            output_tokens.append(self.unk_token)
            if return_offset:
                token_offsets.append([offset, offset + len(chars)])
                return output_tokens,  token_offsets
            else:
                return output_tokens

        is_bad = False
        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                is_bad = True
                break
            sub_tokens.append(cur_substr)
            length = end - start
            if return_offset:
                token_offsets.append([offset, offset + length])
                offset += length
            start = end

        if is_bad:
            output_tokens.append(self.unk_token)
            length = len(chars) - start
            if return_offset:
                token_offsets.append([offset, offset + length])
                offset += length
        else:
            output_tokens.extend(sub_tokens)

        if return_offset:
            return output_tokens, token_offsets
        else:
            return output_tokens

if __name__ == '__main__':
    import time
    text = '易建联是广东男篮的标志性人物，转会可能为零\n在率领球队夺取2018-2019赛季cba总冠军、个人斩获总决赛mvp之后，阿联的职业生涯仿佛迎来了一个新的巅峰。'
    newberttokenizer = NewBertTokenizer.from_pretrained('../../pretrained_models/chinese-roberta-wwm-ext', add_special_tokens = False, do_lower_case = True)
    tokenizer = BertTokenizer.from_pretrained('../../pretrained_models/chinese-roberta-wwm-ext', add_special_tokens = False, do_lower_case = True)
    t0 = time.clock()
    tokens = newberttokenizer(text)
    t1 = time.clock()
    offsets = newberttokenizer.get_offset_mappings(text)
    t2 = time.clock()
    tokens2 = [tokenizer(char)["input_ids"][1] for char in text]
    t3 = time.clock()
    print(len(tokens))
    print(len(offsets))
    print(len(tokens2))
    print(tokens)
    print(offsets)
    print(tokens2)

    print(t1-t0)
    print(t2-t1)
    print(t3-t2)
    print(text[20:21])
    