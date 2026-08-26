from peft import LoraConfig

class LORAConfig(LoraConfig):
    def __init__(self, r:int, lora_alpha:int, target_modules:list, lora_dropout:float):
        super().__init__(r=r, lora_alpha=lora_alpha, \
            target_modules=target_modules, lora_dropout=lora_dropout)
