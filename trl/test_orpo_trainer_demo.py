# This code is built on top of the example code from Huggingface TRL Team
# Available at https://github.com/huggingface/trl/blob/main/examples/scripts/orpo.py

from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from trl import ORPOConfig, ORPOTrainer


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(
        default="microsoft/phi-2", metadata={"help": "the model name"}
    )
    optim: Optional[str] = field(
        default="adamw_torch", metadata={"help": "the model name"}
    )
    data_name: Optional[str] = field(
        default="argilla/ultrafeedback-binarized-preferences-cleaned",
        metadata={"help": "the model name"},
    )
    cache_dir: Optional[str] = field(default="", metadata={"help": "the model name"})
    log_with: Optional[str] = field(
        default="wandb", metadata={"help": "use 'wandb' to log with wandb"}
    )
    output_dir: Optional[str] = field(
        default="", metadata={"help": "use 'wandb' to log with wandb"}
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "the learning rate scheduler"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=4, metadata={"help": "the batch size"}
    )
    num_train_epochs: Optional[int] = field(
        default=5, metadata={"help": "the batch size"}
    )
    beta: Optional[float] = field(
        default=0.25, metadata={"help": "weighting hyperparameter for L_OR"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )


if __name__ == "__main__":
    set_seed(42)
    tqdm.pandas()

    parser = HfArgumentParser(ScriptArguments)  # type: ignore
    script_args = parser.parse_args_into_dataclasses()[0]

    config = ORPOConfig(
        output_dir=script_args.output_dir,
        max_prompt_length=1024,
        max_length=2048,
        logging_steps=100,
        save_strategy="no",
        max_completion_length=2048,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        remove_unused_columns=False,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        beta=script_args.beta,
        report_to="wandb",
        num_train_epochs=script_args.num_train_epochs,
        bf16=True,
        do_eval=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        cache_dir=script_args.cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name, cache_dir=script_args.cache_dir
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

    def build_dataset(tokenizer):
        ds_train = load_dataset(
            script_args.data_name, split="train", cache_dir=script_args.cache_dir
        )

        def chat_template_to_text(sample):
            sample["prompt"] = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": item_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for item_prompt in sample["prompt"]
            ]
            sample["chosen"] = [
                item_chosen[1]["content"] for item_chosen in sample["chosen"]
            ]
            sample["rejected"] = [
                item_rejected[1]["content"] for item_rejected in sample["rejected"]
            ]
            return sample

        ds_train = ds_train.map(chat_template_to_text, batched=True, num_proc=8)  # type: ignore

        return ds_train

    train_ds = build_dataset(tokenizer=tokenizer)

    trainer = ORPOTrainer(
        model=model, args=config, tokenizer=tokenizer, train_dataset=train_ds
    )
    trainer.train()

