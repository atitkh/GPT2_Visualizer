# GPT-2 Layers Visualization

![GPT-2 Visualization](assets/intro.png) <!-- Optional: Add a project logo or screenshot -->

## Introduction

This interactive web application leverages Streamlit to provide a comprehensive and visual exploration of GPT-2's transformer architecture. By breaking down each layer and its components, users can gain an in-depth understanding of how GPT-2 processes and generates text.

Whether you're a student, researcher, or AI enthusiast, this tool is designed to demystify the complexities of transformer models, making them accessible and engaging through intuitive visualizations.

## Features

- **Tokenization Visualization:** Input any text and see how GPT-2 tokenizes it into tokens and assigns token IDs. You can also the model's all tokens and their IDs to a CSV file for further analysis.    
- **Embeddings Exploration:** Visualize token and position embeddings, along with their combined representations using tables and heatmaps. You can also view the embeddings in 3D space using PCA to understand their relationships.
- **Transformer Layer Breakdown:** Dive deep into each transformer layer, exploring layer normalization, multi-head self-attention, residual connections, and the feed-forward neural network (MLP) with their weights and biases, as well as intermediate outputs.
- **Text Generation Control:** Adjust parameters such as Temperature, Top-K, Top-P, to influence the text generation process and reduce repetition.
- **Generated Text Display:** Generate text based on input and visualize the generation steps, including token probabilities and selection processes.
  
## Installation

Follow these steps to set up and run the application locally:

### 1. Clone the Repository

```bash
git clone https://github.com/atitkh/GPT2_Visualizer.git
cd GPT2_Visualizer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

## Usage

1. Open the application in your browser by visiting `http://localhost:8501`.
2. Input text in the text box to visualize tokenization and embeddings.
3. Explore the transformer layers by selecting the layer and sub-layer from the sidebar.
4. Adjust the text generation parameters to influence the generated text.
5. View the generated text and the token selection process.
6. Enjoy exploring GPT-2's architecture and text generation process!

## Acknowledgements

This project was inspired by the [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) and [Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/) blog posts by Jay Alammar. The visualizations and explanations provided in these posts were instrumental in creating this tool.

## Contributing

Feel free to contribute to this project by opening issues, proposing new features, or submitting pull requests. Any contributions you make are **greatly appreciated**.