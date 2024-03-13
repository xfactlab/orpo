def default_args(parser):
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--save_dir", default='./saved', type=str)
    parser.add_argument("--data_name", default='HuggingfaceH4/UltraFeedback', type=str)
    parser.add_argument("--model_name", default="gpt2", type=str)

    # Training Arguments
    parser.add_argument("--torch_compile", default=True, type=bool)
    parser.add_argument("--flash_attention_2", action='store_true')
    parser.add_argument("--lr_scheduler_type", default="cosine", type=str)
    parser.add_argument("--optim", default="paged_adamw_32bit", type=str)
    parser.add_argument("--overwrite_output_dir", default=True, type=bool)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--num_proc", default=8, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--per_device_train_batch_size", default=2, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=2, type=int)
    parser.add_argument("--warmup_steps", default=5000, type=int)    
    parser.add_argument("--evaluation_strategy", default='epoch', type=str)
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--save_strategy", default='epoch', type=str)
    parser.add_argument("--prompt_max_length", default=256, type=int)
    parser.add_argument("--response_max_length", default=1024, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)

    # Wandb Configurations
    parser.add_argument("--wandb_entity", default=None, type=str)
    parser.add_argument("--wandb_project_name", default=None, type=str)

    
    args = parser.parse_args()

    return args