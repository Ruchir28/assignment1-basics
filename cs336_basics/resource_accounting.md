Qustion 1:

Consider GPT-2 XL, which has the following configuration:
vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
num_heads : 25
d_ff : 6,400
Suppose we constructed our model using this configuration. How many trainable parameters
would our model have? Assuming each parameter is represented using single-precision floating
point, how much memory is required to just load this model?


Answer:

token embedding lookup table:
    vocab_size * d_model = 50,257 * 1,600 = 80,411,200

transformer block (num_layer times):
    causal multi head attention:
        q_proj: d_model * d_model = 1,600 * 1,600 = 2,560,000
        k_proj: d_model * d_model = 1,600 * 1,600 = 2,560,000
        v_proj: d_model * d_model = 1,600 * 1,600 = 2,560,000
        o_proj: d_model * d_model = 1,600 * 1,600 = 2,560,000
        total_attention: 4 * 2,560,000 = 10,240,000

    swiglu_ffn:
        gate_proj: d_model * d_ff = 1,600 * 6,400 = 10,240,000
        signal_projection: d_model * d_ff = 1,600 * 6,400 = 10,240,000
        down_proj: d_ff * d_model = 6,400 * 1,600 = 10,240,000
        total_ffn: 30,720,000
    
    rope:
        d_k = d_model // num_heads = 1,600 // 25 = 64
        cos and sin cache: 2 * context_length * d_k = 2 * 1,024 * 64 = 131,072

    rms_norm:
        d_model * 2 (as 2 layers per block) = 1,600 * 2 = 3,200

    total_per_transformer_block: 10,240,000 + 30,720,000 + 131,072 + 3,200 = 41,094,272

all_transformer_blocks: 48 * 41,094,272 = 1,972,525,056

final_rms_norm:
    d_model = 1,600

lm_head:
    vocab_size * d_model = 50,257 * 1,600 = 80,411,200

total_parameters:
    80,411,200 + 1,972,525,056 + 1,600 + 80,411,200 = 2,133,349,056

memory_required:
    2,133,349,056 * 4 = 8,533,396,224 bytes = 7.95 GB

(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped
model. How many FLOPs do these matrix multiplies require in total? Assume that our input
sequence has context_length tokens.

Answer:
Note: Only counting matrix multiplications, not element-wise operations like RMS norm or RoPE.
Assuming batch_size = 1 and sequence_length = context_length = 1024.

Per transformer block (repeated 48 times):

    causal multi head attention:
        d_k = d_model // num_heads = 1,600 // 25 = 64
        
        q_proj: (1, 1024, 1600) @ (1600, 1600) = 2 * 1024 * 1600 * 1600 = 5,242,880,000
        k_proj: (1, 1024, 1600) @ (1600, 1600) = 2 * 1024 * 1600 * 1600 = 5,242,880,000
        v_proj: (1, 1024, 1600) @ (1600, 1600) = 2 * 1024 * 1600 * 1600 = 5,242,880,000
        o_proj: (1, 1024, 1600) @ (1600, 1600) = 2 * 1024 * 1600 * 1600 = 5,242,880,000
        
        attention_scores: (1, 25, 1024, 64) @ (1, 25, 64, 1024) = 2 * 25 * 1024 * 64 * 1024 = 3,355,443,200
        attention_output: (1, 25, 1024, 1024) @ (1, 25, 1024, 64) = 2 * 25 * 1024 * 1024 * 64 = 3,355,443,200
        
        total_attention: 20,971,520,000 + 3,355,443,200 + 3,355,443,200 = 27,682,406,400

    swiglu_ffn:
        gate_proj: (1, 1024, 1600) @ (1600, 6400) = 2 * 1024 * 1600 * 6400 = 20,971,520,000
        signal_proj: (1, 1024, 1600) @ (1600, 6400) = 2 * 1024 * 1600 * 6400 = 20,971,520,000
        down_proj: (1, 1024, 6400) @ (6400, 1600) = 2 * 1024 * 6400 * 1600 = 20,971,520,000
        
        total_ffn: 20,971,520,000 + 20,971,520,000 + 20,971,520,000 = 62,914,560,000

    total_per_transformer_block: 27,682,406,400 + 62,914,560,000 = 90,596,966,400

all_transformer_blocks: 48 * 90,596,966,400 = 4,348,654,387,200

lm_head:
    (1, 1024, 1600) @ (1600, 50257) = 2 * 1024 * 1600 * 50257 = 164,682,137,600

total_matrix_multiply_flops:
    4,348,654,387,200 + 164,682,137,600 = 4,513,336,524,800 FLOPs = 4.51 TFLOPs



