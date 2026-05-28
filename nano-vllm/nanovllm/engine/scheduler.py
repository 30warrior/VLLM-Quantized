from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = config.max_model_len
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        # 
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            #剩余可用的token数量
            if remaining == 0:
                break
            if not seq.block_table: #表示是新序列
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                # 这个序列还需要的token数量
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            #如果剩余可用token数量小于当前序列需要的token数量，且已经调度了序列，则停止调度
            if remaining < num_tokens and scheduled_seqs:  # 只有第一个序列可以分块处理，其他序列必须完整处理或等待下一批
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining) # 当前序列本次调度的token数量（分块预填充 (Chunked Prefill)）
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                #也就是说这个序列的prompt已经全部处理好了，可以进入running队列
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop()) #抢占式调度：内存不足时抢占最老的序列
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1 #decode每次只生成一个token
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # 更新缓存与调度状态
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            if seq.num_tokens >= self.max_model_len:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

'''
示例：处理两个序列

# 序列1: "Hello, how are you?" (5 tokens, 生成10个token)
# 序列2: "What's the weather like?" (6 tokens, 生成8个token)

# 初始状态
waiting = [seq1, seq2]
running = []

# 第1轮: Prefill seq1
scheduled_seqs = [seq1], is_prefill = True
# seq1: 处理5个prompt token，生成1个token
waiting = [seq2]
running = [seq1]

# 第2轮: Prefill seq2  
scheduled_seqs = [seq2], is_prefill = True
# seq2: 处理6个prompt token，生成1个token
waiting = []
running = [seq1, seq2]

# 第3轮: Decode seq1和seq2
scheduled_seqs = [seq1, seq2], is_prefill = False
# seq1: 生成第2个token
# seq2: 生成第2个token
running = [seq1, seq2]

# 第4轮: 继续Decode
scheduled_seqs = [seq1, seq2], is_prefill = False
# seq1: 生成第3个token
# seq2: 生成第3个token
# ... 直到所有序列完成

传统方法:
Batch 1: [seq1_prefill, seq2_prefill]
Batch 2: [seq1_decode]
Batch 3: [seq1_decode, seq2_decode]

Nano-vLLM方法:

Batch 1: [seq1_prefill]           # seq1开始
Batch 2: [seq2_prefill]           # seq2开始，seq1继续
Batch 3: [seq1_decode, seq2_decode] # 两者并行解码

prefill时使用分块预填充，每次只分配当前块需要的KV cache，内存使用更平稳，避免峰值
'''