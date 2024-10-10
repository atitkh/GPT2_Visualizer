import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd  # Ensure pandas is imported
import plotly.express as px  # Ensure plotly is imported
import math
import torch.nn.functional as F
import plotly.graph_objs as go
from bertviz import head_view

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        output_attentions=True,
        output_hidden_states=True
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Set pad_token_id if not already set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

st.title("GPT-2 Steps Visualization")

# User input
st.write(" ### **Input Text**")
user_input = st.text_area("Enter your text here:", "This is a test sentence.")

if user_input:
    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt")
    input_ids = inputs['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    num_tokens_to_generate = st.slider("Number of Tokens to Generate", 1, 50, 4)

    with st.expander("View Tokens and Token IDs"):
        st.write("# **1. Tokenization**")
        token_df = pd.DataFrame({
            "Token": tokens,
            "Token ID": input_ids[0].tolist()
        })
        st.table(token_df)

    with st.expander("View Embeddings"):
        st.write("# **2. Embeddings**")

        # Get token embeddings
        with torch.no_grad():
            token_embeddings = model.transformer.wte(input_ids)[0]  # Shape: (seq_len, embedding_dim)

        # Get position embeddings
        with torch.no_grad():
            position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)  # Shape: (1, seq_len)
            position_embeddings = model.transformer.wpe(position_ids)[0]  # Shape: (seq_len, embedding_dim)

        # Combine token embeddings and position embeddings
        combined_embeddings = token_embeddings + position_embeddings  # Shape: (seq_len, embedding_dim)

        # Display embedding dimensions
        embedding_dim = token_embeddings.shape[1]
        st.write(f"**Embedding Dimension:** {embedding_dim}")

        # For display purposes, show only the first N dimensions
        N = embedding_dim  # Number of dimensions to display

        # Token Embeddings
        token_embeddings_np = token_embeddings.numpy()[:, :N]  # Shape: (seq_len, N)
        token_embeddings_df = pd.DataFrame(token_embeddings_np, index=tokens, columns=[f"Dim {i+1}" for i in range(N)])
        st.write(f"**Token Embeddings (showing {N} dimensions):**")
        st.dataframe(token_embeddings_df.style.format("{:.4f}"))

        # Position Embeddings
        position_embeddings_np = position_embeddings.numpy()[:, :N]  # Shape: (seq_len, N)
        position_indices = [f"Position {i}" for i in range(input_ids.size(1))]
        position_embeddings_df = pd.DataFrame(position_embeddings_np, index=position_indices, columns=[f"Dim {i+1}" for i in range(N)])
        st.write(f"**Position Embeddings (showing {N} dimensions):**")
        st.dataframe(position_embeddings_df.style.format("{:.4f}"))


        # Combined Embeddings
        combined_embeddings_np = combined_embeddings.numpy()[:, :N]  # Shape: (seq_len, N)
        combined_embeddings_df = pd.DataFrame(combined_embeddings_np, index=tokens, columns=[f"Dim {i+1}" for i in range(N)])
        st.write(f"**Combined Embeddings (showing {N} dimensions):**")
        # formula for combined embeddings
        st.latex(r'''
                combined\_embeddings = token\_embeddings + position\_embeddings
                ''')
        st.dataframe(combined_embeddings_df.style.format("{:.4f}"))

        # Visualizing embeddings using PCA

        # Reduce dimensionality for visualization using PCA
        if combined_embeddings.shape[0] >= 2:
            st.write("**Visualizing Embeddings using PCA:**")
            st.write("**2D:**")

            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(token_embeddings_np)
            fig, ax = plt.subplots()
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

            for i, token in enumerate(tokens):
                ax.annotate(token, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

            st.pyplot(fig)
        else:
            st.write("**Not enough tokens to visualize embeddings.**")

        if combined_embeddings.shape[0] >= 3:
            # Visualizing embeddings using PCA 3D

            # Reduce dimensionality for visualization using PCA
            pca = PCA(n_components=3)
            embeddings_3d = pca.fit_transform(token_embeddings_np)

            # Create a 3D scatter plot using Plotly

            fig = go.Figure(data=[go.Scatter3d(
                x=embeddings_3d[:, 0],
                y=embeddings_3d[:, 1],
                z=embeddings_3d[:, 2],
                mode='markers+text',
                text=tokens,
                textposition='top center',
                marker=dict(
                    size=5,
                    color=np.linspace(0, 1, len(tokens)),
                    colorscale='Viridis',
                    opacity=0.8
                )
            )])

            fig.update_layout(
                title='3D:',
                scene=dict(
                    xaxis_title='PC 1',
                    yaxis_title='PC 2',
                    zaxis_title='PC 3'
                )
            )

            st.plotly_chart(fig)

    with st.expander("View Transformer Layer"):
        st.write("# **3. Transformer Layer Breakdown**")

        st.write("""
        In GPT-2, each transformer layer processes the input through several steps:

        1. **Layer Normalization**: Normalizes the input state.
        2. **Multi-Head Self-Attention**: Computes attention weights to focus on different parts of the sequence.
        3. **First Residual Connection**: Adds the attention output back to the input state.
        4. **Layer Normalization**: Normalizes the result after the first residual connection.
        5. **Feed-Forward Neural Network (MLP)**: Applies non-linear transformations to each position separately.
        6. **Second Residual Connection**: Adds the MLP output back to the input of the MLP.

        """)

        # Get outputs with hidden states and attentions
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
        attentions = outputs.attentions  # Tuple of (num_layers, batch_size, num_heads, seq_len, seq_len)
        hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1, batch_size, seq_len, hidden_size)

        num_layers = len(attentions)
        num_heads = attentions[0].size(1)

        layer = st.slider("Select Transformer Layer", 1, num_layers, 1)
        head = st.slider("Select Attention Head", 1, num_heads, 1)

        # Extract the hidden states for the selected layer and the previous layer
        hidden_states_prev = hidden_states[layer - 1][0]  # Input to the layer
        hidden_states_layer = hidden_states[layer][0]     # Output of the layer
        
        # Create unique token labels
        token_labels = [f"{token}_{i}" for i, token in enumerate(tokens)]

        # Get the transformer block for the selected layer
        transformer_block = model.transformer.h[layer - 1]
        ln_1 = transformer_block.ln_1
        attn = transformer_block.attn
        ln_2 = transformer_block.ln_2
        mlp = transformer_block.mlp

        # Get model configuration
        hidden_size = model.config.hidden_size  # Should be 768
        num_attention_heads = model.config.num_attention_heads  # Should be 12
        head_dim = hidden_size // num_attention_heads  # Should be 64

        st.write(f"**State Before Layer {layer} Input:** Shape {tuple(hidden_states_prev.shape)}")
        hidden_states_prev_df = pd.DataFrame(hidden_states_prev.detach().numpy(), index=token_labels, columns=[f"Dim {i+1}" for i in range(hidden_size)])
        st.dataframe(hidden_states_prev_df.style.format("{:.4f}"))

        st.write(f"**Model Configuration:**")
        st.write(f"- Model Dimension Size: {hidden_size}")
        st.write(f"- Number of Layers: {num_layers}")
        st.write(f"- Number of Attention Heads: {num_attention_heads}")
        st.write(f"- Head Dimension: {head_dim}")

        ### **Step 1: Layer Normalization (Before Attention)**
        st.write("## **Step 1: Layer Normalization (Before Attention Input)**")

        with torch.no_grad():
            # Apply Layer Normalization
            hidden_states_norm = ln_1(hidden_states_prev.unsqueeze(0))[0]  # Shape: (seq_len, hidden_size)

        st.write(f"**Layer-Normalized Output:** Shape {tuple(hidden_states_norm.shape)}")
        st.latex(r''' \text{Layer-Normalized Output } H_{\text{norm}} = \gamma \odot \frac{Input - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta ''')
        # show gamma and beta values when i hover over the layer normalization
        st.write("**Where:**")
        ln_1_gamma = ln_1.weight.detach().numpy()
        ln_1_beta = ln_1.bias.detach().numpy()
        ln_1_gamma_df = pd.DataFrame(ln_1_gamma, columns=["Gamma"])
        ln_1_beta_df = pd.DataFrame(ln_1_beta, columns=["Beta"])

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**γ (Scaling Factor):**")
            st.dataframe(ln_1_gamma_df.style.format("{:.4f}"), height=100)

        with col2:
            st.write(f"**β (Shift):**")
            st.dataframe(ln_1_beta_df.style.format("{:.4f}"), height=100)
            
        hidden_states_norm_df = pd.DataFrame(hidden_states_norm.numpy(), index=token_labels, columns=[f"Dim {i+1}" for i in range(hidden_size)])
        st.dataframe(hidden_states_norm_df.style.format("{:.4f}"))

        ### **Step 2: Multi-Head Self-Attention**

        st.write("## **Step 2: Multi-Head Self-Attention (12 Heads)**")

        with torch.no_grad():
            # The weight matrix for c_attn is (hidden_size, 3 * hidden_size)
            W = attn.c_attn.weight  # Shape: (hidden_size, 3 * hidden_size)
            b = attn.c_attn.bias    # Shape: (3 * hidden_size,)

            # Compute QKV
            qkv = torch.matmul(hidden_states_norm, W) + b  # Shape: (seq_len, 3 * hidden_size)

            # Split into Q, K, V
            Q, K, V = qkv.split(hidden_size, dim=-1)  # Each has shape: (seq_len, hidden_size)

            # Reshape Q, K, V to (seq_len, num_heads, head_dim)
            Q = Q.view(-1, num_attention_heads, head_dim)
            K = K.view(-1, num_attention_heads, head_dim)
            V = V.view(-1, num_attention_heads, head_dim)

            # Select the Q, K, V for the chosen head
            selected_head = head - 1  # Zero-indexed
            Q_head = Q[:, selected_head, :]  # Shape: (seq_len, head_dim)
            K_head = K[:, selected_head, :]
            V_head = V[:, selected_head, :]

        # Display Q, K, V for the selected head with weight and bias
        st.write("**Here:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.latex(r'W_q')
            W_q_df = pd.DataFrame(
                W[:, :head_dim].detach().numpy(),
                columns=[f"Dim {i+1}" for i in range(head_dim)]
            )
            st.dataframe(W_q_df.style.format("{:.4f}"), height=200, width=400)

            st.latex(r'b_q')
            b_q_df = pd.DataFrame(b[:head_dim].detach().numpy(), columns=["Bias"])
            st.dataframe(b_q_df.style.format("{:.4f}"), height=100)

        with col2:
            st.latex(r'W_k')
            W_k_df = pd.DataFrame(
                W[:, head_dim:2*head_dim].detach().numpy(),
                columns=[f"Dim {i+1}" for i in range(head_dim)]
            )
            st.dataframe(W_k_df.style.format("{:.4f}"), height=200, width=400)

            st.latex(r'b_k')
            b_k_df = pd.DataFrame(b[head_dim:2*head_dim].detach().numpy(), columns=["Bias"])
            st.dataframe(b_k_df.style.format("{:.4f}"), height=100)

        with col3:
            st.latex(r'W_v')
            W_v_df = pd.DataFrame(
                W[:, 2*head_dim:3*head_dim].detach().numpy(),
                columns=[f"Dim {i+1}" for i in range(head_dim)]
            )
            st.dataframe(W_v_df.style.format("{:.4f}"), height=200, width=400)

            st.latex(r'b_v')
            b_v_df = pd.DataFrame(b[2*head_dim:3*head_dim].detach().numpy(), columns=["Bias"])
            st.dataframe(b_v_df.style.format("{:.4f}"), height=100)

        st.write(f"**Queries (Q)**")
        st.latex(r'''Q = H_{\text{norm}} \times W_Q + b_Q''')
        st.write(f"**Queries (Q) for Layer {layer}, Head {head}:** Shape {tuple(Q_head.shape)}")
        Q_df = pd.DataFrame(Q_head.numpy(), index=token_labels, columns=[f"Dim {i+1}" for i in range(head_dim)])
        st.dataframe(Q_df.style.format("{:.4f}"))

        st.write(f"**Keys (K)**")
        st.latex(r'''K = H_{\text{norm}} \times W_K + b_K''')
        st.write(f"**Keys (K) for Layer {layer}, Head {head}:** Shape {tuple(K_head.shape)}")
        K_df = pd.DataFrame(K_head.numpy(), index=token_labels, columns=[f"Dim {i+1}" for i in range(head_dim)])
        st.dataframe(K_df.style.format("{:.4f}"))

        st.write(f"**Values (V)**")
        st.latex(r'''V = H_{\text{norm}} \times W_V + b_V''')
        st.write(f"**Values (V) for Layer {layer}, Head {head}:** Shape {tuple(V_head.shape)}")
        V_df = pd.DataFrame(V_head.numpy(), index=token_labels, columns=[f"Dim {i+1}" for i in range(head_dim)])
        st.dataframe(V_df.style.format("{:.4f}"))

        ### **Step 3: Scaled Dot-Product Attention**

        st.write("## **Step 3: Dot-Product Attention**")

        st.latex(r'''\text{Attention\_Scores} = \frac{Q \cdot K^T}{\sqrt{d_k}}''')

        with torch.no_grad():
            # Compute attention scores
            attention_scores = torch.matmul(Q_head, K_head.transpose(0, 1)) / math.sqrt(head_dim)  # Shape: (seq_len, seq_len)

            # Apply softmax to get attention weights
            attention_weights = attentions[layer - 1][0][head - 1].detach()  # Shape: (seq_len, seq_len)

        # Display attention scores and weights
        st.write(f"**Attention Scores for Layer {layer}, Head {head}:** Shape {tuple(attention_scores.shape)}")
        attention_scores_df = pd.DataFrame(attention_scores.numpy(), index=token_labels, columns=token_labels)
        st.dataframe(attention_scores_df.style.format("{:.4f}"))

        # show masking of attention scores
        st.write(f"**Attention Scores with Masking for Layer {layer}, Head {head}:**")
        st.latex(r''' \text{Masked\_Attention\_Scores}_{i,j} = \begin{cases} \text{Attention\_Scores}_{i,j} & \text{if } j \leq i \\ -\infty & \text{otherwise} \end{cases} ''')

        attention_scores_masked = attention_scores.clone()
        attention_scores_masked[torch.triu(torch.ones_like(attention_scores_masked), diagonal=1) == 1] = float('-inf')
        attention_scores_masked_df = pd.DataFrame(attention_scores_masked.numpy(), index=token_labels, columns=token_labels)
        st.dataframe(attention_scores_masked_df.style.format("{:.4f}"))

        st.write(f"**Attention Weights for Layer {layer}, Head {head}:** Shape {tuple(attention_weights.shape)}")
        st.latex(r'''\text{Attention\_Weights}_{i,j} = \frac{\exp(\text{Masked\_Attention\_Scores}_{i,j})}{\sum_{k=1}^{n} \exp(\text{Masked\_Attention\_Scores}_{i,k})}''')
        attention_weights_df = pd.DataFrame(attention_weights.numpy(), index=token_labels, columns=token_labels)
        st.dataframe(attention_weights_df.style.format("{:.4f}"))

        ### **Visualizing Attention Weights**

        st.write("### **Visualizing Attention Weights**")

        show_all_heads = st.checkbox("Show All Heads", value=False)

        if show_all_heads:
            fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
            for i, ax in enumerate(axes.flat):
                sns.heatmap(attentions[layer - 1][0][i].detach().numpy(), ax=ax, cmap="viridis", annot=True, fmt=".2f",
                            xticklabels=token_labels,
                            yticklabels=token_labels)
                ax.set_title(f"Layer {layer}, Head {i+1}")
            st.pyplot(fig)

        else:
            fig, ax = plt.subplots()
            sns.heatmap(attention_weights.numpy(), xticklabels=token_labels, yticklabels=token_labels, cmap='viridis', ax=ax, annot=True)
            ax.set_title(f"Layer {layer}, Head {head}")
            ax.set_xlabel("Key Tokens")
            ax.set_ylabel("Query Tokens")
            st.pyplot(fig)

        # fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        # for i, ax in enumerate(axes.flat):
        #     sns.heatmap(attentions[layer - 1][0][i].detach().numpy(), ax=ax, cmap="viridis", annot=True, fmt=".2f",
        #                 xticklabels=tokenizer.convert_ids_to_tokens(input_ids[0]),
        #                 yticklabels=tokenizer.convert_ids_to_tokens(input_ids[0]))
        #     ax.set_title(f"Head {i+1}")
        # st.pyplot(fig)

        ### **Step 4: Attention Output**

        st.write("## **Step 4: Attention Output**")

        st.latex(r'''\text{Attention\_Output} = \text{Attention\_Weights} \times V''')

        with torch.no_grad():
            attention_output_head = torch.matmul(attention_weights, V_head)  # Shape: (seq_len, head_dim)

        st.write(f"**Attention Output for Layer {layer}, Head {head}:** Shape {tuple(attention_output_head.shape)}")
        attention_output_df = pd.DataFrame(attention_output_head.numpy(), index=token_labels, columns=[f"Dim {i+1}" for i in range(head_dim)])
        st.dataframe(attention_output_df.style.format("{:.4f}"))

        ### **Step 5: Concatenate Heads and Linear Projection**

        st.write("## **Step 5: Combine Heads and Linear Projection**")

        st.write("""
        The attention outputs from all heads are concatenated and passed through a linear layer (`c_proj`) which changes the vector's dimension.
        """)

        with torch.no_grad():
            # Compute attention outputs for all heads
            attention_outputs = []
            for h in range(num_attention_heads):
                Q_h = Q[:, h, :]  # Shape: (seq_len, head_dim)
                K_h = K[:, h, :]
                V_h = V[:, h, :]

                # Compute attention scores and weights
                scores_h = torch.matmul(Q_h, K_h.transpose(0, 1)) / math.sqrt(head_dim)
                weights_h = attentions[layer - 1][0][h].detach()  # Shape: (seq_len, seq_len)

                # Compute attention output
                output_h = torch.matmul(weights_h, V_h)  # Shape: (seq_len, head_dim)
                attention_outputs.append(output_h)

            # Concatenate attention outputs from all heads
            attention_output_concat = torch.cat(attention_outputs, dim=-1)  # Shape: (seq_len, hidden_size)

            # Linear projection
            attn_output = attn.c_proj(attention_output_concat)  # Shape: (seq_len, hidden_size)

        st.write(f"**Concatenated Attention Output:** Shape {tuple(attention_output_concat.shape)}")
        st.latex(r''' \text{Attention\_Output\_Concat} = \text{Concatenate}(\text{Attention\_Output\_Head}_1, \ldots, \text{Attention\_Output\_Head}_{\text{num\_heads}}) ''')
        attention_output_concat_df = pd.DataFrame(attention_output_concat.numpy(), index=token_labels, columns=[f"Dim {i+1}" for i in range(hidden_size)])
        st.dataframe(attention_output_concat_df.style.format("{:.4f}"))

        st.write(f"**Linear Projection**")
        st.latex(r''' \text{Attention\_Output} = \text{Attention\_Output\_Concat} \times W_{\text{c\_proj\_1}} + b_{\text{c\_proj\_1}} ''')
        
        st.write(f"**Where:**")
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r'''W_{\text{c\_proj\_1}}''')
            st.write(f"**Weight is not visualizable due to its large size of {tuple(attn.c_proj.weight.shape)}**")

        with col2:
            st.latex(r'''b_{\text{c\_proj\_1}}''')
            b_c_proj_1_df = pd.DataFrame(attn.c_proj.bias.detach().numpy(), columns=["Bias"])
            st.dataframe(b_c_proj_1_df.style.format("{:.4f}"), height=100)

        st.write(f"**Attention Output After Projection:** Shape {tuple(attn_output.shape)}")
        attn_output_df = pd.DataFrame(attn_output.numpy(), index=token_labels, columns=[f"Dim {i+1}" for i in range(hidden_size)])
        st.dataframe(attn_output_df.style.format("{:.4f}"))

        ### **Step 6: First Residual Connection**

        st.write("## **Step 6: First Residual Connection**")

        with torch.no_grad():
            # Residual connection
            residual_1 = hidden_states_prev + attn_output  # Shape: (seq_len, hidden_size)
            
        st.write(f"**Output After First Residual Connection:** Shape {tuple(residual_1.shape)}")
        st.latex(r''' \text{Residual\_1} = \text{Attention\_Input} + \text{Attention\_Output} ''')
        residual_1_df = pd.DataFrame(residual_1.numpy(), index=token_labels, columns=[f"Dim {i+1}" for i in range(hidden_size)])
        st.dataframe(residual_1_df.style.format("{:.4f}"))


        ### **Step 7: Layer Normalization (Before MLP)**

        st.write("## **Step 7: Layer Normalization (Before MLP)**")

        with torch.no_grad():
            # Layer normalization
            hidden_states_norm_2 = ln_2(residual_1.unsqueeze(0))[0]  # Shape: (seq_len, hidden_size)

        st.latex(r'''\text{Layer-Normalized Output }H_{\text{norm}} = \gamma \odot \frac{\text{Residual\_1} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta ''')
        st.write("Where:")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**γ (Scaling Factor):**")
            ln_2_gamma = ln_2.weight.detach().numpy()
            ln_2_gamma_df = pd.DataFrame(ln_2_gamma, columns=["Gamma"])
            st.dataframe(ln_2_gamma_df.style.format("{:.4f}"), height=100)

        with col2:
            st.write(f"**β (Shift):**")
            ln_2_beta = ln_2.bias.detach().numpy()
            ln_2_beta_df = pd.DataFrame(ln_2_beta, columns=["Beta"])
            st.dataframe(ln_2_beta_df.style.format("{:.4f}"), height=100)

        st.write(f"**Layer-Normalized Output Before MLP:** Shape {tuple(hidden_states_norm_2.shape)}")
        hidden_states_norm_2_df = pd.DataFrame(hidden_states_norm_2.numpy(), index=token_labels, columns=[f"Dim {i+1}" for i in range(hidden_size)])
        st.dataframe(hidden_states_norm_2_df.style.format("{:.4f}"))

        ### **Step 8: Feed-Forward Neural Network (MLP)**

        st.write("## **Step 8: Feed-Forward Neural Network (MLP)**")

        st.write("### ***It consists of two linear layers with a GELU activation in between:***")

        with torch.no_grad():
            first_linear = mlp.c_fc(hidden_states_norm_2.unsqueeze(0))[0]  # Shape: (seq_len, 4 * hidden_size)
            gelu_activation = F.gelu(first_linear)
            # MLP output
            mlp_output = mlp(hidden_states_norm_2.unsqueeze(0))[0]  # Shape: (seq_len, hidden_size)

        st.latex(r''' \text{First\_Linear} = H_{\text{norm}} \times W_{\text{c\_fc}} + b_{\text{c\_fc}} ''')
        st.write("Where:")
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r'''W_{\text{c\_fc}}''')
            st.write(f"**Weight is not visualizable due to its large size of {tuple(mlp.c_fc.weight.shape)}**")

        with col2:
            st.latex(r'''b_{\text{c\_fc}}''')
            b_c_fc = mlp.c_fc.bias.detach().numpy()
            b_c_fc_df = pd.DataFrame(b_c_fc, columns=["Bias"])
            st.dataframe(b_c_fc_df.style.format("{:.4f}"), height=100)

        st.write(f"**First Linear Layer Output:** Shape {tuple(first_linear.shape)}")
        first_linear_df = pd.DataFrame(first_linear.numpy(), index=token_labels, columns=[f"Dim {i+1}" for i in range(4 * hidden_size)])
        st.dataframe(first_linear_df.style.format("{:.4f}"))

        st.write(f"**GELU Activation Output:** Shape {tuple(gelu_activation.shape)}")
        st.latex(r''' GELU_{\text{fl}}= 0.5 \times (1 + \text{erf}(\frac{\text{First\_Linear}}{\sqrt{2}})) \times \text{First\_Linear}''')
        gelu_activation_df = pd.DataFrame(gelu_activation.numpy(), index=token_labels, columns=[f"Dim {i+1}" for i in range(4 * hidden_size)])
        st.dataframe(gelu_activation_df.style.format("{:.4f}"))

        st.write(f"**MLP Output:** Shape {tuple(mlp_output.shape)}")
        st.latex(r''' \text{MLP\_Output} = GELU_{\text{fl}} \times W_{\text{c\_proj\_2}} + b_{\text{c\_proj\_2}} ''')
        st.write("Where:")
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r'''W_{\text{c\_proj\_2}}''')
            st.write(f"**Weight is not visualizable due to its large size of {tuple(mlp.c_proj.weight.shape)}**")

        with col2:
            st.latex(r'''b_{\text{c\_proj\_2}}''')
            b_c_proj_2 = mlp.c_proj.bias.detach().numpy()
            b_c_proj_2_df = pd.DataFrame(b_c_proj_2, columns=["Bias"])
            st.dataframe(b_c_proj_2_df.style.format("{:.4f}"), height=100)

        mlp_output_df = pd.DataFrame(mlp_output.numpy(), index=token_labels, columns=[f"Dim {i+1}" for i in range(hidden_size)])
        st.dataframe(mlp_output_df.style.format("{:.4f}"))

        ### **Step 9: Residual Connection**

        st.write("## **Step 9: Residual Connection**")

        with torch.no_grad():
            # Second residual connection
            output_layer = residual_1 + mlp_output  # Shape: (seq_len, hidden_size)

        st.write(f"**Output of Transformer Layer {layer}:** Shape {tuple(output_layer.shape)}")
        st.latex(r''' \text{Residual\_2} = \text{Residual\_1} + \text{MLP\_Output} ''')
        output_layer_df = pd.DataFrame(hidden_states_layer.detach().numpy(), index=token_labels, columns=[f"Dim {i+1}" for i in range(hidden_size)])
        st.dataframe(output_layer_df.style.format("{:.4f}"))

    with st.expander("View Probabilities"):
        st.write("# **4. Text Generation**")

        do_sample = st.checkbox("Use Sampling", value=False)
        # Get the output of the selected layer
        st.write(f"**Logits Output:**")
        st.latex(r'''
                    \text{Logits} = \text{Residual\_2} \times W_{\text{unemb}}
                    ''')

        unembedded_output = torch.matmul(output_layer, model.transformer.wte.weight.T)
        
        st.write(f"**Here:**")
        st.write(f"**Weight and Output is not visualizable due to its large size of {tuple(model.transformer.wte.weight.T.shape)}**")

        # Convert logits to probabilities
        st.write("**Logits to Probabilities:**")
        probabilities = torch.softmax(unembedded_output, dim=-1)  # Shape: (seq_len, vocab_size)
        st.latex(r'''
            P(\text{next token}) = \frac{\exp(\text{Logits}_i)}{\sum_{j} \exp(\text{Logits}_j)}
        ''')

        if do_sample:
            temperature = st.slider("Temperature", 0.0, 1.5, 0.7)
            top_k = st.slider("Top-k", 0, 100, 50)
            top_p = st.slider("Top-p", 0.0, 1.0, 0.90)
        else:
            temperature = 1.0
            top_k = None
            top_p = None

        generated_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=inputs['attention_mask'],
            max_length=input_ids.size(1) + num_tokens_to_generate,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
            early_stopping=True,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id
        )

        generated_sequence = generated_outputs.sequences[0]
        generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)

        # Get the scores (logits) for each generated token
        scores = generated_outputs.scores  # List of tensors, each of shape (batch_size, vocab_size)
        # The generated tokens (excluding the input tokens)
        generated_tokens = generated_sequence[input_ids.size(1):]

        for idx, (logits, token_id) in enumerate(zip(scores, generated_tokens)):
            # logits: (batch_size, vocab_size)
            st.write(f" ### **Step {idx + 1}: Predicting Token {idx + 1}**")

            # Convert logits to probabilities
            probabilities = torch.softmax(logits[0], dim=0)  # logits[0] since batch_size=1

            # Get top 10 tokens
            top_k_probs = 10
            top_probs, top_indices = torch.topk(probabilities, top_k_probs)
            top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())
            top_token_ids = top_indices.tolist()

            # Get the actual generated token
            generated_token = tokenizer.convert_ids_to_tokens([token_id.item()])[0]
            generated_token_id = token_id.item()

            st.write(f"**Generated Token {idx + 1}: '{generated_token}' (Token ID: {generated_token_id})**")

            # Prepare data for plotting
            prob_data = {
                "Token": top_tokens,
                "Token ID": top_token_ids,
                "Probability": top_probs.detach().numpy()
            }

            # Create a bar chart with hover data including Token ID
            fig = px.bar(
                prob_data,
                x='Token',
                y='Probability',
                hover_data={'Token ID': True, 'Token': True, 'Probability': ':.4f'},
                labels={'Token': 'Token', 'Probability': 'Probability'},
                title=f'Top {top_k_probs} Next Token Probabilities for Token {idx + 1}'
            )

            st.plotly_chart(fig)

    st.write("### **Generated Text:**")
    st.write(generated_text)

    st.subheader("Summary")

    # Create a summary table
    pipeline_data = []

    # Input Text
    pipeline_data.append({
        "Step": "Input Text",
        "Value": user_input
    })

    # Tokenization
    token_str = ', '.join([f"'{token}' ({token_id})" for token, token_id in zip(tokens, input_ids[0].tolist())])
    pipeline_data.append({
        "Step": "Tokenization",
        "Value": token_str
    })

    # Embeddings
    pipeline_data.append({
        "Step": "Embeddings",
        "Value": f"Generated by adding token and position embeddings"
    })

    # Transformer Layers
    pipeline_data.append({
        "Step": "Transformer Layers",
        "Value": f"Processed through {len(attentions)} layers"
    })

    # Generated Text
    pipeline_data.append({
        "Step": "Generated Text",
        "Value": generated_text
    })

    # Convert to DataFrame
    pipeline_df = pd.DataFrame(pipeline_data)

    st.table(pipeline_df)

# footer text
st.markdown("""
    <div style="text-align: center; margin-top: 50px; bottom: 0;">

    <p><strong>Created by:</strong> <a href="https://www.github.com/atitkh" target="_blank">Atit Kharel</a></p>

    </div>
""", unsafe_allow_html=True)
