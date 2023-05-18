# Deduplication groups evaluator

This tool is a set of tests on different Graph Neural Networkds for the activity of the deduplication groups evaluation.
The purpose of the network is to evaluate the goodness (in terms of percentage - 0% bad group, 100% perfect group) of a group created by an automatic algorithm for the disambiguation of entity.
The code presents the use case of Author Name Disambiguation, where groups are created by a pairwise comparison stage and a final close mesh stage to put all the equivalent authors in the same group of entities.

The input dataset is obtained by deduplicating using the FDup framework (https://peerj.com/articles/cs-1058/) on a subset of pubmed records extracted from the OpenAIRE Research Graph (https://graph.openaire.eu/).
Such publications have been processed using a custom LDA model on the abstract and Authors have been extracted by each one of them to create a new entity with attributes inherited by the publication itself (e.g. a new author is identified by the publication identifier, the LDA topics vector of his publication, and the co-authors in the same publication).
The deduplication of such authors is based on a preliminary LNFI (Last Name First Initial) clustering stage to limit the number of comparisons, followed by a decision tree on their attributes.
Two authors are considered to be equivalent if they share at least 2 co-authors and/or they have a cosine similarity between the LDA topics vectors greater than 0.5.
Once the close mesh stage have been done, the following features have been extracted to feed the Neural Network with meaningful information:
- a pretrained BERT model described to extract 768-sized feature vectors from the abstract
- a custom encoder (so-called Bag Of Letters) to extract 55-sized feature vectors for the author name

The dataset resulting from this procedure is available at: *LINK TO THE DATASET (to-be-defined)*

The code in this release tests 3 different base architectures:
- 3-layered Graph Convolutional Network (GCN)
- 3-layered Graph Attention Network (GAT)
- 6-layered Graphormer Network (SmallGraphormer)

Once the most promising direction have been identified in the GAT, the model is further customized by adding an LSTM layer and by considering edge weights, author names encoding and node weights.
The best model is proved to be the GAT3NamesEdgesCentralityLSTM with ~89% of accuracy on the test set.

The entire code in this release have been developed using PyTorch (https://pytorch.org/) and the DGL library for Graph Neural Networks (https://www.dgl.ai/), while results of the experiments have been visualized using Tensorboard (https://www.tensorflow.org/tensorboard).