import logging

import torch


class GPT2ForCausalLM:
    def __init__(self,model,config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.model.to(self.device).half()   # 模型与缓存同设备同 dtype(FP16)
        self.wte = self.model.transformer.wte #Embedding vector
        self.spe = self.model.transformer.wpe #position Embedding
        self.layers = self.model.transformer.h #layers
        self.num_layers = len(self.layers)
        self.logger = logging.getLogger(__name__)


        # === 从 model.config 读模型结构相关字段 === 
        cfg = self.model.config 
        self.dim = cfg.n_embd # 768  (hidden size) 
        self.num_heads = cfg.n_head # 12   (attention heads) 
        self.head_dim = self .dim // self .num_heads # 64 
        # GPT2 是 MHA 不是 GQA，num_kv_heads == num_heads 
        self.num_kv_heads = self.num_heads # 12 
        self.vocab_size  = cfg.vocab_size # 50257 
        self.num_layers  = cfg.n_layer # 12   (跟 len(layers) 一致)

        # === KV 缓存配置 ===
        self.num_blocks = config.num_blocks
        self.block_size = config.block_size
        self.dtype = torch.float16

        #shape: [num_blocks,block_size,num_head,head_dim]
        self.k_tensor,self.v_tensor = self.init_real_memory()

    def forward(self, inputs, block_table: list[int], slot: list[int], start_pos: int = 0):
        """
        前向传播
        inputs: [batch_size, list[input_id]]
        block_table / slot: 由 Scheduler 侧 BlockManager 记账后传入的索引元数据,
        模型层不记账,只按 slot 读写物理 KV 池
        start_pos: 本步输入在完整序列中的起始位置,决定位置编码偏移与 KV 读回长度
        ln1 -> attn -> residual -> ln2 -> mlp -> residual -> output
        output: [batch, seq_len, 768]
        """
        hidden_states = self._embedding(inputs, start_pos)   # [batch, seq_len, 768]
        batch, seq_len = hidden_states.shape[0], hidden_states.shape[1]

        for i in range(self.num_layers):
            layer = self.layers[i]

            # === Attention 子层 ===
            residual = hidden_states                       # 备份原始输入做残差
            attn_input = layer.ln_1(hidden_states)         # LN1（不覆盖 hidden_states）

            qkv = layer.attn.c_attn(attn_input)             # [batch, seq, 2304]
            q, k, v = qkv.split(self.dim, dim=-1)           # 各 [batch, seq, 768]

            # 拆头: [batch, seq, 768] -> [batch, seq, H, D] -> [batch, H, seq, D]
            q = q.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            # 写 KV 到缓存
            self._write_to_cache(k, v, slot, i)

            # 读历史 KV（覆盖 start_pos + 本步新 token 的完整上下文）
            k_all, v_all = self._find_past_kv(block_table, i, num_tokens=start_pos + seq_len)
            if k_all is None:
                k_all, v_all = k, v   # 兜底：直接用当前 K/V

            # attention（prefill: q 和 k 长度相同，内部自动加 causal mask）
            attn_out = self._attention(q, k_all, v_all)          # [batch, H, seq, D]
            attn_out = self._process_attn_output(attn_out, i)     # [batch, seq, 768]
            hidden_states = residual + attn_out                   # 残差 1

            # === MLP 子层 ===
            residual = hidden_states
            mlp_input = layer.ln_2(hidden_states)                # LN2
            mlp_out = self._MLP(mlp_input, i)
            hidden_states = residual + mlp_out                    # 残差 2

        logits = self.model.lm_head(hidden_states)
    
        return logits

    def _embedding(self,inputs:list[list[int]],start_pos:int = 0):
        """
        获取token的embedding并叠加位置编码
        return hidden state [batch_size,len(input_id),hidden_dim]
        """
        input_id = torch.as_tensor(inputs, device=self.device)
        seq_len = input_id.shape[-1]

        #位置编码
        #position是一个1dimension的tensor = [start_pos,start_pos+1,...,start_pos+seq_pos+seq_len-1]
        positions = torch.arange(start_pos, start_pos + seq_len, device=self.device)
        return self.wte(input_id) + self.spe(positions).unsqueeze(0)
    
    def _map_qkv(self,hidden_states,layer_index: int):
        """
        hidden_states: [batch_size,len(input_id),num_heads*head_dim,hidden_dim] -> hidden_dim = num_heads*head_dim
        return: [batch_size,len(input_id),num_heads,3*dim]
        """
        return self.layers[layer_index].attn.c_attn(hidden_states)
    
    def _write_to_cache(self, k_states, v_states, slot, layer_idx):
        """
        写入缓存
        k_states and v_states : [batch_size, num_kv_heads, num_new_token, head_dim]
        slot : [batch_size * num_new_token] 每个 token 在缓存中的位置 (block_id * block_size + offset)
        layer_idx : layer index
        """

        #check the shape of k_states and v_states
        if k_states.shape != v_states.shape:
            raise ValueError("k_states and v_states must have the same shape")

        k_cache = self.k_tensor[layer_idx]
        v_cache = self.v_tensor[layer_idx]
        k_flat = k_cache.view(-1, self.num_kv_heads, self.head_dim)
        v_flat = v_cache.view(-1, self.num_kv_heads, self.head_dim)

        """
        这里的k_new和v_new将输入先交换位置，前两个是batch_size, num_new_token,后面的维度展开
        """
        k_new = k_states.permute(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_dim)
        v_new = v_states.permute(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_dim)
        for i in range(len(slot)):
            k_flat[slot[i]] = k_new[i]
            v_flat[slot[i]] = v_new[i]
    
    def _find_past_kv(self,seq_blocks: list[int],layer_Index: int,num_tokens: int | None  = None):
        """
        查找历史缓存
        return k,v : [batch_size,num_kv_heads, seq_len, head_dim]
        num_token:作为替代slot的参数，其在sequence没有填满block的时候起作用，可以让我们减少多余复制
        """
        block_list = seq_blocks

        if block_list == {}:
            return None,None


        """
        这里把两个list中的view合并，复制到一个新的缓存地址
        """    
        k_past= []
        v_past = []
        for block in block_list:
            k_past.append(self.k_tensor[layer_Index][block])
            v_past.append(self.v_tensor[layer_Index][block])
        """
        这里把两个list中的view合并，复制到一个新的缓存地址
        """    
        k = torch.cat(k_past,dim = 0)
        v = torch.cat(v_past,dim = 0)

        if num_tokens is not None:
            k = k[:num_tokens]
            v = v[:num_tokens]
        
        # [seq_len, H, D] -> [H, seq_len, D]  -> [1, H, seq_len, D]
        k = k.permute(1,0,2).unsqueeze(0)
        v = v.permute(1,0,2).unsqueeze(0)
        
        return k,v
    
    def _attention(self,query,keys,values):
        """
        query = [batch_size,num_head,1,head_dim]
        keys = [batch_size,num_kv_heads,seq_len,head_dim]
        values = [batch_size,num_kv_heads,num_head,seq_len,head_dim]
        return: [batch_size,num_head,seq_q,head_dim]
        """

        seq_q = query.shape[2]
        seq_len = keys.shape[2]
        is_prefill = seq_len == seq_q

        # K^T
        k = keys.transpose(-1,-2)

        #Q*K^T
        #此时采用批量矩阵乘法，结果形状：[H,seq_q,seq_k]
        qk = torch.matmul(query,k)

        #缩放
        scaling = self.head_dim**-0.5
        qk = qk * scaling

        #prefill的时候加causal mask：query token i 不能看 key token： j>i
        if is_prefill:
            mask = torch.tril(torch.ones(seq_q, seq_len, dtype=torch. bool , device=qk.device))
            qk = qk.masked_fill(~mask, float ( '-inf' ))

        #softmax
        attn_weights = torch.softmax(qk,dim = -1) #让每个seq_q对每个seq_k的注意力权重归一化

        #计算注意力输出
        #[batch_size,H,seq_q,seq_k] * [batch_size,H,seq_k,head_dim] -> [batch_size,H,seq_q,head_dim]
        attn_output = torch.matmul(attn_weights,values)

        return attn_output

    
    def _process_attn_output(self,attention_output,layer_index):
        """
        attention_output = [batch_size,num_head,,seq_q,head_dim]
        """
        layer = self.layers[layer_index]
        attention_output = attention_output.permute(0,2,1,3).reshape(-1,self.dim) #合并head
        attention_output = layer.attn.c_proj(attention_output) #将上一行强行合并的multi_head进行一次线性变换，融合。
        
        return attention_output
    
    def _residual(self,hidden_states,residual):
        """
        这里需要接受两次residual：
            1. hidden_states + atten_out -> [N,(num_heads*head_dim)] -> 矩阵里每个元素相加，不改变个数
            2. h1 + mlp_out -> [N,(num_heads*head_dim)] -> 
        """

        return hidden_states + residual
    
    
    def _MLP(self,h1,layer_index):
        """
        一次MLP被我们理解成一次潜亏网络，对每个token独立地做一次升维->非线性激活->降维
        """
        layer = self.layers[layer_index]

        #MLP
        mlp_output = layer.mlp(h1) #[N,768]

        return mlp_output
    

    def init_real_memory(self):
        k_tensors = []
        v_tensors = []
        # 初始化 KV 缓存
        # kv_tensor 代表每个 layer 对应的所有缓存
        for layer in range(self.num_layers):
            try:
                self.logger.info(f"Try to init the layer {layer} CUDA memory...")
                k_tensor = torch.empty((self.num_blocks,self.block_size,self.num_kv_heads,self.head_dim), 
                dtype=self.dtype,
                device = self.device,
                out = None,
                layout = None,
                requires_grad = False)
                v_tensor = torch.empty((self.num_blocks,self.block_size,self.num_kv_heads,self.head_dim), 
                dtype=self.dtype,
                device = self.device,
                out = None,
                layout = None,
                requires_grad = False)

            except Exception as e:
                self.logger.error(f"KV 缓存初始化失败：{e}")
                raise

            self.logger.info(f"Layer {layer} K 缓存初始化完成，大小为：{k_tensor.shape}，dtype为：{k_tensor.dtype},device:{k_tensor.device},dim:{k_tensor.dim()},总元素数：{k_tensor.numel()}")
            self.logger.info(f"Layer {layer} 每个 block 元素数为：{k_tensor.numel() / self.block_size}，总共的字节数：{k_tensor.numel() * k_tensor.dtype.itemsize} bytes")
            self.logger.info(f"Layer {layer} V 缓存初始化完成，大小为：{v_tensor.shape}，dtype为：{v_tensor.dtype},device:{v_tensor.device},dim:{v_tensor.dim()},总元素数：{v_tensor.numel()}")          
            self.logger.info(f"Layer {layer} 每个 block 元素数为：{v_tensor.numel() / self.block_size}，总共的字节数：{v_tensor.numel() * v_tensor.dtype.itemsize} bytes") 
            k_tensors.append(k_tensor)
            v_tensors.append(v_tensor)

        return k_tensors, v_tensors