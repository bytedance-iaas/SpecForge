



import tiktoken
from tiktoken.load import dump_tiktoken_bpe
from transformers.integrations.tiktoken import convert_tiktoken_to_fast
import json

# 方法1: 如果 tiktoken.model 是标准格式，尝试直接加载
def load_kimi_encoding():
    try:
        # 尝试从文件直接创建编码
        model_path = "/root/models/Kimi-K2-Instruct/tiktoken.model"
        
        # 读取二进制文件并尝试解析为 BPE ranks
        with open(model_path, 'rb') as f:
            data = f.read()
        
        # 尝试使用 tiktoken 的内部加载函数
        from tiktoken.load import load_tiktoken_bpe
        mergeable_ranks = load_tiktoken_bpe(model_path)
        
    except:
        # 如果失败，使用基础编码 + 特殊 token
        print("无法直接加载 tiktoken.model，使用基础编码...")
        base_encoding = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = base_encoding._mergeable_ranks
    
    # Kimi 的特殊 token
    special_tokens = {
        "[BOS]": 163584,
        "[EOS]": 163585,
        "<|im_end|>": 163586,
        "<|im_user|>": 163587,
        "<|im_assistant|>": 163588,
        "<|start_header_id|>": 163590,
        "<|end_header_id|>": 163591,
        "[EOT]": 163593,
        "<|im_system|>": 163594,
        "<|tool_calls_section_begin|>": 163595,
        "<|tool_calls_section_end|>": 163596,
        "<|tool_call_begin|>": 163597,
        "<|tool_call_argument_begin|>": 163598,
        "<|tool_call_end|>": 163599,
        "<|im_middle|>": 163601,
        "[UNK]": 163838,
        "[PAD]": 163839
    }
    
    # 创建 tiktoken.Encoding 对象
    encoding = tiktoken.Encoding(
        name="kimi_k2",
        pat_str=r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?[\p{L}]+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens
    )
    
    return encoding

# 方法2: 从已有的 tokenizer 提取编码对象
def extract_encoding_from_tokenizer():
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("/root/models/Kimi-K2-Instruct", trust_remote_code=True)
    
    # 检查是否有内部的 tiktoken 编码对象
    if hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'encoding'):
        return tokenizer.tokenizer.encoding
    elif hasattr(tokenizer, '_tiktoken_tokenizer'):
        return tokenizer._tiktoken_tokenizer
    else:
        return None

# 执行转换
try:
    # 先尝试从 tokenizer 提取
    encoding = extract_encoding_from_tokenizer()
    
    if encoding is None:
        # 如果提取失败，手动创建
        encoding = load_kimi_encoding()
    
    print(f"成功创建编码对象: {encoding.name}")
    print(f"特殊 token 数量: {len(encoding._special_tokens)}")
    print(f"BPE ranks 数量: {len(encoding._mergeable_ranks)}")
    
    # 转换为 fast tokenizer
    convert_tiktoken_to_fast(encoding, "/root/models/Kimi-K2-Instruct")
    print("转换完成！")
    
    # 验证转换结果
    from transformers import AutoTokenizer
    fast_tokenizer = AutoTokenizer.from_pretrained("/root/models/Kimi-K2-Instruct")
    print(f"验证: {fast_tokenizer.is_fast}")
    
except Exception as e:
    print(f"转换失败: {e}")
    import traceback
    traceback.print_exc()


# # 验证转换结果
# from transformers import AutoTokenizer, PreTrainedTokenizerFast
# fast_tokenizer = PreTrainedTokenizerFast.from_pretrained("/root/models/Kimi-K2-Instruct", use_fast=True, trust_remote_code=True)

# print(f"验证: {fast_tokenizer.is_fast}")