import logging

from miniserving.Backend.kvCache.KVPool import KVPool


class KVCacheManager:
    def __init__(self,config):
      self.logger = logging.getLogger(__name__)
      self.KvPool = KVPool(config)

      #slot mapping: 下一个token 写入的位置
      #Metadate context_lens,query lens, seq_lens
    


    
