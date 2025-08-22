# MGT Model

This is the source code for paper [Help Me Screen: Analyzing and Predicting the Success of Start-ups in Dynamic Venture Capital Networks](xxx).

## Abstract
Most start-ups fail, and early-stage ventures face even lower survival rates. Identifying high-potential start-ups remains a critical challenge for venture capital (VC) investors and policymakers. While predictive models exist, the evolving relationships between VC investors, start-ups, and management teams in dynamic networks are underexplored. We propose a method to predict whether a start-up will succeed within five years of its first funding round. Using a 40-year global VC dataset, we model the VC ecosystem as a dynamic bipartite network linking start-ups to individuals (investors/managers). Our approach incrementally updates graph embeddings through unsupervised self-attention to incorporate new nodes, edges, and their neighbors. Node embeddings are further fine-tuned via link prediction and classification tasks, while temporal dependencies are captured to form sequential representations. The model identifies early-stage start-ups with twice the success likelihood of those chosen by professional investors. Key factors including networking and education align with VC literature. Additionally, we provide model complexity analysis and open-source our implementation to support practical applications and future research.

## Data

## Prerequisites
The code has been successfully tested in the following environment. (For older PyG versions, you may need to modify the code)
- Python 3.8.12
- PyTorch 1.11.0
- Pytorch Geometric 2.0.4
- Sklearn 1.0.2
- Pandas 1.3.5

## Getting Started

### Prepare your data

We provide samples of our data in the `./Data` folder. The input of our model is as follows:

* `graph_edges` includes the edges of each time step. The shape is [Time_num x 2 x Edge_num]. Time_num is the number of time steps. Edge_num is the number of the edge in this time step.
* `edge_date` is the time step corresponding to each edge and the length is equal to the number of all edges.
* `edge_type` is the edge type corresponding to each edge and the length is equal to the number of all edges.
* `all_nodes` is the number of nodes.
* `new_companies` is the index of the newly added node at each time. The shape is [(Time_num - 1) x new_add_node_length].
* `labels` is the label of the newly added node at each time. The shape is [(Time_num - 1 ) x new_add_node_length].
* `nodetypes` is the set of node types corresponding to all nodes.

**Node Representation Learning**

`node_representation_learning.py` : File for generating node representations in VC networks by node classification and link prediction tasks

```python
python node_representation_learning.py --embedding_dim 64 --n_layers_clf 3 --train_embed --loss_type 'LPNC'
```

**Start-up Success Prediction**

`startup_success_prediction.py` : Code that dynamically updates newly added nodes and predicts the success of startups

```python
python startup_success_prediction.py --dynamic_clf True --gpus 'cuda:0'
```

**File Statement**
Run the node_representation_learning.py file to generate the representation of the nodes and save the embedding in the file `Save_model`. Then run the startup_success_prediction.py file to make predictions about the success of the startups.
Model/Convs.py contains **MGTConvs**, which is the layer to update the nodes dynamically. **Predict_model** in `Model/Model.py` is the model for startup success prediction.

## Cite

Please cite our paper if you find this code useful for your research:

```
citation
```


