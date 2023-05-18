model_config = {
    # The type of model (gpt2, gpt3, gpt4, etc.)
    'model_type': 'gpt4',

    # The size of the model (small, medium, large, etc.)
    'model_size': 'large',

    # The number of layers in the model
    'num_layers': 24,

    # The number of attention heads in the model
    'num_attention_heads': 16,

    # The hidden size of the model
    'hidden_size': 1024,

    # The number of tokens in the vocabulary
    'vocab_size': 50257,

    # The maximum sequence length
    'max_seq_length': 1024,

    # The number of tokens to predict
    'num_tokens_to_predict': 20,

    # The number of samples to generate
    'num_samples': 1,

    # The temperature for sampling
    'temperature': 1.0,

    # The top-k value for sampling
    'top_k': 0,

    # The top-p value for sampling
    'top_p': 0.9,

    # The number of GPUs to use
    'num_gpus': 1,

    # The number of CPU threads to use
    'num_cpu_threads': 4,

    # The batch size
    'batch_size': 1,

    # The learning rate
    'learning_rate': 0.0001,

    # The number of training steps
    'num_train_steps': 10000,

    # The number of warmup steps
    'num_warmup_steps': 1000,

    # The optimizer to use
    'optimizer': 'Adam',

    # The epsilon value for Adam
    'epsilon': 1e-08,

    # The weight decay
    'weight_decay': 0.01,

    # The gradient clipping value
    'gradient_clipping': 1.0,

    # The learning rate schedule
    'lr_schedule': 'warmup_linear',

    # The seed for the random number generator
    'seed': 42,
}
