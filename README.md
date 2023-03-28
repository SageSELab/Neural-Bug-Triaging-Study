This repo holds the code for our empirical study on using transformer-based LMs for bug triaging.

## Architecture of Fine-Tuning Transformer-based Models

![alt text](https://github.com/SageSELab/Neural-Bug-Triaging-Study/blob/main/Models/Transformer-based%20LM%20Fine-Tuning-1.png)

To fine-tune all transformer-based models, we adhere to the standard sequence classification approach, which involves appending a classification layer on top of transformers' outputs. Given the similarity in the overall architecture of the transformer-based models, we utilize it for fine-tuning. The overall architecture works as follows. Initially, tokenization of the input sequence is done using a tokenizer. Then, a special classification token [CLS] is inserted at the sequence's beginning, followed by all tokens passed through multiple transformer layers. Here, it is worth noting that the number of transformer layers employed in each model distinguishes them from one another, and the hyperparameters' settings vary accordingly. The last transformer layer in these models generates contextualized embeddings for each token, capturing the token's context within the entire sequence. For developer and component classification, only the [CLS] embeddings are considered since it is used for other sequence classification. The [CLS] embedding is commonly known as the aggregated representation of the entire input sequence and is therefore used for sequence classification. Once the [CLS] embedding is obtained from the last layer, it is fed into a softmax classification layer for assigning developers and components to bug reports.

## Folder Structure

Our implementation of each based can be found in the **Models** directory. The following python packages are required for running for our baselines.

- `transformers`
- `torch`
- `sklearn`
- `pandas`
- `numpy`

**Diagrams** directory contains our all diagrams of orthogonality and statistical significance test. **Results** directory contains Top1, Top5, Top5 and MRR of developer assignment, and precision, recall and F1-score of component assignment.
