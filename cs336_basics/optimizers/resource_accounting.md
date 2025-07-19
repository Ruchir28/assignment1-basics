Question:
How much peak memory does running AdamW require? Decompose your answer based on the
memory usage of the parameters, activations, gradients, and optimizer state. Express your answer
in terms of the batch_size and the model hyperparameters (vocab_size, context_length,
num_layers, d_model, num_heads). Assume d_ff = 4 ×d_model.
For simplicity, when calculating memory usage of activations, consider only the following compo-
nents:
• Transformer block
    – RMSNorm(s)
    – Multi-head self-attention sublayer: QKV projections, Q⊤ K matrix multiply, softmax,
    weighted sum of values, output projection.
    – Position-wise feed-forward: W1 matrix multiply, SiLU, W2 matrix multiply
• final RMSNorm
• output embedding
• cross-entropy on logits

Deliverable: An algebraic expression for each of parameters, activations, gradients, and opti-
mizer state, as well as the total.

Answer:

Given:

- B: batch_size
- L: context_length
- D: d_model
- V: vocab_size
- N: num_layers
- H: num_heads
- d_ff: 4 * D (as given)
- N_T : number of transformer blocks 

1. Model Parameters (Weights)

    - Token Embedding Layer
        Shape : (V * D)
        parameters: (V * D)

    - RoPE
        cos,sin cached: L * D/2 * 2 
        paramters: L * D

    - Transformer block (N_T)
        
        - QKV Projections:Three separate weight matrices W_Q, W_K, W_V to create Q, K, and V.
            Shape (D,D)
            paramter: 3 * D^2

        - OutPut Projections
            Shape (D,D)
            parameter: D^2
        
        - Swiglu Feed Forward Network
            - gate,signal,down_proj matrices each of weights d_ff * D -> 4D^2
            parameters: 3*4D^2 = 12D^2

        - RMSNorm(2 per transformer color)
            parameters: D * 2 

    - Final Layer Norm
        parameters: D 

    - Output Embedding
        Shape : (D * V)
    parameters: DV 

    Total parameters for model:  VD + LD + N_T*(3D^2 + D^2 + 12D^2 + 2D) + D + DV 

2. Gradients: 
    Same as model parameters 

3. Optimizer states
    2 * model_parameters

4. Activations

- Transformer Blocks
    - RMSNorm output
        Shape same as input : (B * L * D)
    - Q,K,V Projections
        3 * B * L * D
    - Attention scores (attention for each head is calculated seperately)
        Q^T @ K -> B * H * L * L 
    - Softmax (same as prev input)
        B * H * L * L
    - Conxtexutal embedding (multiplying by V)
        B * L * D
    - Output projection
        B * L * D
    - Siluffn Layer Gate 
        B * L * d_ff -> B * L * 4D
    - Siluffn Layer singal 
        B * L * d_ff -> B * L * 4D
    - Siluffn GATE * SIGNAL
        B * L * 4D
    - Silu Down Proj
        B * L * D
- Final RmsNorm Block
    B * L * D
- Output Embedding 
    B * L * V
    





    