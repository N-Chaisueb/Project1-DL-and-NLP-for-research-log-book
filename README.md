# Project: Implementation of deep learning and NLP in research log book for text generation

## Context and Problem Statement
- Research log book is a day to day record of raw data, methods, measurements, and experimental results obtained during research activities.
- The research log book occupies more than 100 pages within a few months.
- To analyze or summarize a specific task in the research log book is time-consuming.
- Therefore, this text generation project was performed to support researchers to easily review their work, identify insight, and receive conclusions from the research log book.
- The script will take words/phrases as an input and predict the next words/sentences relating to the input words based on textual data in the research log book.
- Transformer and GPT-2 (Generative Pre-trained Transformers) models have been used to train textual dataset and generate new text.
- Firstly, the textual data was converted to numerical data by using NLP techniques such as Tokenization, Vectorization, and Padding.
- The sparse categorical cross-entropy was defined to compute the cross-entropy loss between the labels and predictions because the labels are represented by the index of the category and there are a lot of categories, like text data. Each label refers to each word in the text file.
- Regularization techniques consisting of dropout layer, batch normalization layer, and early stopping were used to address overfitting issues in the neural network.
- Custom callbacks of monitoring validation loss, early stopping, and learning rate scheduling were determined for improving model performance.

## Data source
- The research log book was recorded at Methodology Light Source (MLS), Germany.
- MLS is an electron circular accelerator providing synchrotron radiation in terahertz (THz) to extreme ultraviolet (EUV) regime.
- Textual dataset includes date and time, researcher names, experiments, results, issues, and conclusion.

## Models
### 1. Transformer model
- Transformer model is a type of deep learning model that can capture long-range dependencies between input data in a sequence by using self-attention mechanisms.
- The self-attention mechanisms allow the transformer model to attend to all positions in the input sequence simultaneously to capture long-range dependencies and focus on relevant parts of the input sequence. That means they provide a parallel computation.
- The transformer model is split into 2 parts, an encoder and a decoder parts. The encoder operates on each word of the input sequence and projects them into query, key, and value vectors, while the decoder operates on each word of the target sequence and uses vectors from the encoder part to produce outputs. 
- For natural language processing (NLP) tasks, the transformer model can be used for translation, text summarization, and question answering.
- However, training a transformer model requires lots of textual data and computational resources. Therefore, we can use transfer learning techniques to apply pre-trained models and fine-tune these models to our specific task.

### 2. GPT-2 model (Generative Pre-trained Transformers)
- GPT-2 model, originally invented by OpenAI, is a pre-trained Large Language Model (LLM) based on the Transformer architecture. It is trained on massive amounts of text all around the internet with next word prediction purposes.
- The core of Generative LLMs is to predict the next word in a sentence and to generate coherent text based on user prompts. It can be applied to various natural language processing (NLP) tasks, such as text generation, question answering, and machine translation.
- In this project, the pre-trained GPT-2 model was fine-tuned to a specific text style and the textual data recorded in the research log book.

## Process
1. Data import
2. Text Preprocessing >>> Tokenization, Vectorization, Padding and Creating sequences of text data 
3. Custom callbacks >>> Monitoring validation loss, Early stopping, and Learning rate scheduling
4. Transformer model
   - Embedding layer
   - Positional encoding
   - Multiple encoder blocks >>> Multi-head attention layer, Feed-forward neural network layer, Residual Connection, and Layer Normalization
   - Multiple decoder blocks >>> Masked multi-head attention layer, Multi-head attention layer, Feed-forward neural network layer, Residual Connection, and Layer Normalization
6. Text generation from trained transformer model
7. Fine-tuned GPT-2 model with unfreezing layers
8. Text generation from trained GPT-2 model
9. Model evaluation and Visualization of training performance with accuracy and loss
