
import os
import time
import wandb
import torch
import argparse
from datasets import load_dataset
from typing import List, Dict, Union
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from src.args import default_args
from src.orpo_trainer import ORPOTrainer
from src.utils import preprocess_logits_for_metrics

class ORPO(object):
    def __init__(self, args) -> None:
        self.start = time.gmtime()
        self.args = args

        # Load Tokenizer
        print(">>> 1. Loading Tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, cache_dir=self.args.cache_dir)
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
            print("     1-1. Chat Template Applied (<|user|> <|assistant|>)")
        else:
            pass
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load Model
        print(">>> 2. Loading Model")
        if self.args.flash_attention_2:
            self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name, 
                                                              cache_dir=self.args.cache_dir,
                                                              torch_dtype=torch.bfloat16,
                                                              attn_implementation="flash_attention_2")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name, 
                                                              cache_dir=self.args.cache_dir,
                                                              torch_dtype=torch.bfloat16)
                                                          
        # Load Dataset
        print(">>> 3. Loading Dataset")
        self.data = load_dataset(self.args.data_name, cache_dir=self.args.cache_dir)

        # Preprocess Dataset
        print(">>> 4. Filtering and Preprocessing Dataset")
        train_split = "train_prefs" if 'binarized' in self.args.data_name.lower() else "train"
        test_split = "test_prefs" if 'binarized' in self.args.data_name.lower() else "test"

        train = self.data[train_split].filter(self.filter_dataset)
        test = self.data[test_split].filter(self.filter_dataset)
        print(f"\n\n>>> {len(train)} / {len(self.data[train_split])} rows left after filtering by prompt length.")

        self.train = train.map(self.preprocess_dataset, batched=True, num_proc=self.args.num_proc, remove_columns=self.data[train_split].column_names)
        self.test = test.map(self.preprocess_dataset, batched=True, num_proc=self.args.num_proc, remove_columns=self.data[test_split].column_names)                              
            
        # Set WANDB & Logging Configurations
        self.run_name = f"{self.args.model_name.split('/')[-1]}-{self.args.data_name.split('/')[-1]}-ORPO-{self.start.tm_mday}-{self.start.tm_hour}-{self.start.tm_min}"
        self.save_dir = os.path.join('./checkpoints/', f"{self.args.data_name.split('/')[-1]}/{self.run_name}")
        self.log_dir = os.path.join('./checkpoints/', f"{self.args.data_name.split('/')[-1]}/{self.run_name}/logs")
        
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def preprocess_dataset(self, examples: Union[List, Dict]):
        prompt = [self.tokenizer.apply_chat_template([item[0]], tokenize=False, add_generation_prompt=True) for item in examples['chosen']]
        chosen = [self.tokenizer.apply_chat_template(item, tokenize=False) for item in examples['chosen']]
        rejected = [self.tokenizer.apply_chat_template(item, tokenize=False) for item in examples['rejected']]
    
        model_inputs = self.tokenizer(prompt,
                                      max_length=self.args.response_max_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors='pt')
        pos_labels = self.tokenizer(chosen,
                                    max_length=self.args.response_max_length,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors='pt')
        neg_labels = self.tokenizer(rejected,
                                    max_length=self.args.response_max_length,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors='pt') 
                
        model_inputs['positive_input_ids'] = pos_labels['input_ids']
        model_inputs['positive_attention_mask'] = pos_labels['attention_mask']
        
        model_inputs['negative_input_ids'] = neg_labels['input_ids']
        model_inputs['negative_attention_mask'] = neg_labels['attention_mask']
        
        return model_inputs

    def filter_dataset(self, examples: Union[List, Dict]):
        if self.tokenizer.apply_chat_template([examples['chosen'][0]], tokenize=True, add_generation_prompt=True, return_tensors='pt').size(-1) < self.args.prompt_max_length:    
            return True
        else:
            return False

    def prepare_trainer(self):
        wandb.init(name=self.run_name)
        arguments = TrainingArguments(
            torch_compile=self.args.torch_compile,
            output_dir=self.save_dir,  # The output directory
            logging_dir=self.log_dir,
            logging_steps=50,
            learning_rate=self.args.lr,
            overwrite_output_dir=True,  # overwrite the content of the output directory
            num_train_epochs=self.args.num_train_epochs,  # number of training epochs
            per_device_train_batch_size=self.args.per_device_train_batch_size,  # batch size for training
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,  # batch size for evaluation
            evaluation_strategy=self.args.evaluation_strategy,  # batch size for evaluation
            save_strategy=self.args.evaluation_strategy,
            optim=self.args.optim,
            warmup_steps=self.args.warmup_steps,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            gradient_checkpointing=True, #if ('llama' in self.args.model_name.lower()) or ('mistral' in self.args.model_name.lower()) else False,
            gradient_checkpointing_kwargs={'use_reentrant':True},
            load_best_model_at_end=True,
            do_train=True,
            do_eval= True,
            lr_scheduler_type=self.args.lr_scheduler_type,
            remove_unused_columns=False,
            report_to='wandb',
            run_name=self.run_name,
            bf16=True
        )
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        
        self.trainer = ORPOTrainer(
            model=self.model,
            alpha=self.args.alpha,
            pad=self.tokenizer.pad_token_id,
            args=arguments,
            train_dataset=self.train,
            eval_dataset=self.test,
            data_collator=data_collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        
    def run(self):
        print(">>> 5. Preparing ORPOTrainer")
        self.prepare_trainer()
        self.trainer.train()

        # Saving code for FSDP
        if self.trainer.is_fsdp_enabled:
            self.trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        self.trainer.save_model()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser("ORPO")
    args = default_args(parser)

    # Set WANDB configurations
    if args.wandb_entity is not None and args.wandb_project_name is not None:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
        os.environ["WANDB_PROJECT"] = args.wandb_project_name
    else:
        pass
    os.environ["TOKENIZERS_PARALLELISM"] = 'false'

    print("================================================================================================\n")
    print(f">>> Fine-tuning {args.model_name} with ORPO on {args.data_name}\n")
    print("================================================================================================")
    print("\n\n>>> Summary:")
    print(f"    - Lambda              : {args.alpha}")
    print(f"    - Training Epochs     : {args.num_train_epochs}")
    print(f"    - Prompt Max Length   : {args.prompt_max_length}")
    print(f"    - Response Max Length : {args.response_max_length}")

    item = ORPO(args=args)
    item.run()
