"""Module providing all the models used in the paper."""
import torch
from torch import nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn

torch.manual_seed(42)

# GRAPH CONVOLUTIONAL NETWORKS
class GCN3(nn.Module):
    """Class for 3-layered GCN"""
    def __init__(self, num_features, hidden_dim, dropout):
        super().__init__()
        self.dropout = dropout
        self.conv1 = dglnn.GraphConv(num_features, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv3 = dglnn.GraphConv(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, g, nfeats):
        """Forward function"""
        # Apply graph convolution and activation.
        hidden_states = F.relu(self.conv1(g, nfeats))
        hidden_states = F.relu(self.conv2(g, hidden_states))
        hidden_states = F.relu(self.conv3(g, hidden_states))
        with g.local_scope():
            g.ndata["h"] = hidden_states
            hidden_global = dgl.mean_nodes(g, "h")
            hidden_global = F.dropout(hidden_global, p=self.dropout, training=self.training)
            return F.sigmoid(self.classifier(hidden_global))


class GCN3WeightedEdges(nn.Module):
    """Class for 3-layered GCN with weighted edges"""
    def __init__(self, num_features, hidden_dim, dropout):
        super().__init__()
        self.dropout = dropout

        self.norm = dglnn.EdgeWeightNorm(norm="left")

        self.conv1 = dglnn.GraphConv(num_features, hidden_dim, norm="none")
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, norm="none")
        self.conv3 = dglnn.GraphConv(hidden_dim, hidden_dim, norm="none")

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, g, nfeats):
        """Forward function"""
        norm_edge_weights = self.norm(g, g.edata["weights"])
        # Apply graph convolution and activation.
        hidden_states = F.relu(self.conv1(g, nfeats, edge_weight=norm_edge_weights))
        hidden_states = F.relu(self.conv2(g, hidden_states, edge_weight=norm_edge_weights))
        hidden_states = F.relu(self.conv3(g, hidden_states, edge_weight=norm_edge_weights))
        with g.local_scope():
            g.ndata["h"] = hidden_states
            hidden_global = dgl.mean_nodes(g, "h")
            hidden_global = F.dropout(hidden_global, p=self.dropout, training=self.training)
            return F.sigmoid(self.classifier(hidden_global))


class GCN3Names(nn.Module):
    """Class for 3-layered GCN with names encoding"""
    def __init__(self, num_features, hidden_dim, dropout):
        super().__init__()
        self.dropout = dropout
        self.conv1 = dglnn.GraphConv(num_features, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv3 = dglnn.GraphConv(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, g, nfeats):
        """Forward function"""
        # Apply graph convolution and activation.
        hidden_states = F.relu(self.conv1(g, torch.cat((nfeats, g.ndata["names"]), dim=1)))
        hidden_states = F.relu(self.conv2(g, hidden_states))
        hidden_states = F.relu(self.conv3(g, hidden_states))
        with g.local_scope():
            g.ndata["h"] = hidden_states
            hidden_global = dgl.mean_nodes(g, "h")
            hidden_global = F.dropout(hidden_global, p=self.dropout, training=self.training)
            return F.sigmoid(self.classifier(hidden_global))


# GRAPH ATTENTION NETWORKS
class GAT3(nn.Module):
    """Class for 3-layered GAT"""
    def __init__(self, num_features, hidden_dim, num_heads, dropout, feat_drop, attn_drop):
        super().__init__()
        self.dropout = dropout
        self.conv1 = dglnn.GATConv(num_features, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)
        self.conv2 = dglnn.GATConv(hidden_dim, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)
        self.conv3 = dglnn.GATConv(hidden_dim, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, g, nfeats):
        """Forward function"""
        # Apply graph convolution and activation.
        hidden_states = F.relu(self.conv1(g, nfeats))
        hidden_states = torch.mean(hidden_states, dim=1)
        hidden_states = F.relu(self.conv2(g, hidden_states))
        hidden_states = torch.mean(hidden_states, dim=1)
        hidden_states = F.relu(self.conv3(g, hidden_states))
        hidden_states = torch.mean(hidden_states, dim=1)

        with g.local_scope():
            g.ndata["h"] = hidden_states
            hidden_global = dgl.mean_nodes(g, "h")
            hidden_global = F.dropout(hidden_global, p=self.dropout, training=self.training)
            return F.sigmoid(self.classifier(hidden_global))


class GAT3LSTM(nn.Module):
    """Class for 3-layered GAT LSTM"""
    def __init__(self, num_features, hidden_dim, num_heads, dropout, feat_drop, attn_drop, num_layers):
        super().__init__()
        self.dropout = dropout
        self.conv1 = dglnn.GATConv(num_features, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)
        self.conv2 = dglnn.GATConv(hidden_dim, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)
        self.conv3 = dglnn.GATConv(hidden_dim, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)

        # global attention for the last layer
        self.global_attention0 = dglnn.GlobalAttentionPooling(gate_nn=nn.Linear(hidden_dim, 1))
        self.global_attention1 = dglnn.GlobalAttentionPooling(gate_nn=nn.Linear(hidden_dim, 1))
        self.global_attention2 = dglnn.GlobalAttentionPooling(gate_nn=nn.Linear(hidden_dim, 1))
        self.global_attention3 = dglnn.GlobalAttentionPooling(gate_nn=nn.Linear(hidden_dim, 1))

        # long short term memory
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=False)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, g, nfeats):
        """Forward function"""
        with g.local_scope():
            hidden_global_0 = self.global_attention0(g, nfeats)
            # Apply graph convolution and activation.
            hidden_states = F.relu(self.conv1(g, nfeats))
            hidden_states = torch.mean(hidden_states, dim=1)
            hidden_global1 = self.global_attention1(g, hidden_states)
            hidden_states = F.relu(self.conv2(g, hidden_states))
            hidden_states = torch.mean(hidden_states, dim=1)
            hidden_global2 = self.global_attention2(g, hidden_states)
            hidden_states = F.relu(self.conv3(g, hidden_states))
            hidden_states = torch.mean(hidden_states, dim=1)
            hidden_global3 = self.global_attention3(g, hidden_states)

            hidden_global = torch.stack((hidden_global_0, hidden_global1, hidden_global2, hidden_global3), dim=0)

            hidden_global, (last_hidden_lstm, _) = self.lstm(hidden_global)
            hidden_global = F.dropout(last_hidden_lstm[-1, :, :], p=self.dropout, training=self.training)
            return F.sigmoid(self.classifier(hidden_global))


class GAT3NamesLSTM(nn.Module):
    """Class for 3-layered GAT with LSTM and names encoding"""
    def __init__(self, num_features, hidden_dim, num_heads, dropout, feat_drop, attn_drop, num_layers):
        super().__init__()
        self.dropout = dropout
        self.conv1 = dglnn.GATConv(num_features, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)
        self.conv2 = dglnn.GATConv(hidden_dim, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)
        self.conv3 = dglnn.GATConv(hidden_dim, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)

        # global attention for the last layer
        self.global_attention0 = dglnn.GlobalAttentionPooling(gate_nn=nn.Linear(hidden_dim, 1))
        self.global_attention1 = dglnn.GlobalAttentionPooling(gate_nn=nn.Linear(hidden_dim, 1))
        self.global_attention2 = dglnn.GlobalAttentionPooling(gate_nn=nn.Linear(hidden_dim, 1))
        self.global_attention3 = dglnn.GlobalAttentionPooling(gate_nn=nn.Linear(hidden_dim, 1))

        # long short term memory
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=False)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, g, nfeats):
        """Forward function"""
        names = g.ndata["names"]
        with g.local_scope():
            hidden_global_0 = self.global_attention0(g, torch.cat((nfeats, names), dim=1))
            # Apply graph convolution and activation.
            hidden_states = F.relu(self.conv1(g, torch.cat((nfeats, names), dim=1)))
            hidden_states = torch.mean(hidden_states, dim=1)
            hidden_global1 = self.global_attention1(g, hidden_states)
            hidden_states = F.relu(self.conv2(g, hidden_states))
            hidden_states = torch.mean(hidden_states, dim=1)
            hidden_global2 = self.global_attention2(g, hidden_states)
            hidden_states = F.relu(self.conv3(g, hidden_states))
            hidden_states = torch.mean(hidden_states, dim=1)
            hidden_global3 = self.global_attention3(g, hidden_states)

            hidden_global = torch.stack((hidden_global_0, hidden_global1, hidden_global2, hidden_global3), dim=0)

            hidden_global, (last_hidden_lstm, _) = self.lstm(hidden_global)
            hidden_global = F.dropout(last_hidden_lstm[-1, :, :], p=self.dropout, training=self.training)
            return F.sigmoid(self.classifier(hidden_global))



class GAT3NamesEdgesLSTM(nn.Module):
    """Class for 3-layered GAT with LSTM, names encoding and weigthed edges"""
    def __init__(self, num_features, hidden_dim, num_heads, dropout, feat_drop, attn_drop, num_layers):
        super().__init__()
        self.dropout = dropout
        self.conv1 = dglnn.GATConv(num_features, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)
        self.conv2 = dglnn.GATConv(hidden_dim, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)
        self.conv3 = dglnn.GATConv(hidden_dim, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)

        # global attention for the last layer
        self.global_attention0 = dglnn.GlobalAttentionPooling(gate_nn=nn.Linear(hidden_dim, 1))
        self.global_attention1 = dglnn.GlobalAttentionPooling(gate_nn=nn.Linear(hidden_dim, 1))
        self.global_attention2 = dglnn.GlobalAttentionPooling(gate_nn=nn.Linear(hidden_dim, 1))
        self.global_attention3 = dglnn.GlobalAttentionPooling(gate_nn=nn.Linear(hidden_dim, 1))

        # long short term memory
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=False)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, g, nfeats):
        """Forward function"""
        names = g.ndata["names"]
        with g.local_scope():
            hidden_global0 = self.global_attention0(g, torch.cat((nfeats, names), dim=1))
            # Apply graph convolution and activation.
            hidden_states = F.relu(self.conv1(g, torch.cat((nfeats, names), dim=1), edge_weight=g.edata["weights"]))
            hidden_states = torch.mean(hidden_states, dim=1)
            hidden_global1 = self.global_attention1(g, hidden_states)
            hidden_states = F.relu(self.conv2(g, hidden_states, edge_weight=g.edata["weights"]))
            hidden_states = torch.mean(hidden_states, dim=1)
            hidden_global2 = self.global_attention2(g, hidden_states)
            hidden_states = F.relu(self.conv3(g, hidden_states, edge_weight=g.edata["weights"]))
            hidden_states = torch.mean(hidden_states, dim=1)
            hidden_global3 = self.global_attention3(g, hidden_states)

            hidden_global = torch.stack((hidden_global0, hidden_global1, hidden_global2, hidden_global3), dim=0)

            hidden_global, (last_hidden_lstm, _) = self.lstm(hidden_global)
            hidden_global = F.dropout(last_hidden_lstm[-1, :, :], p=self.dropout, training=self.training)
            return F.sigmoid(self.classifier(hidden_global))


class GAT3NamesEdgesCentralityLSTM(nn.Module):
    """Class for 3-layered GAT with LSTM, names encoding, weigthed edges and centrality encoding"""
    def __init__(self, num_features, hidden_dim, num_heads, dropout, feat_drop, attn_drop, num_layers):
        super().__init__()
        self.dropout = dropout
        self.conv1 = dglnn.GATConv(num_features, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)
        self.conv2 = dglnn.GATConv(hidden_dim, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)
        self.conv3 = dglnn.GATConv(hidden_dim, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)

        # global attention for the last layer
        self.global_attention0 = dglnn.GlobalAttentionPooling(gate_nn=nn.Linear(hidden_dim, 1))
        self.global_attention1 = dglnn.GlobalAttentionPooling(gate_nn=nn.Linear(hidden_dim, 1))
        self.global_attention2 = dglnn.GlobalAttentionPooling(gate_nn=nn.Linear(hidden_dim, 1))
        self.global_attention3 = dglnn.GlobalAttentionPooling(gate_nn=nn.Linear(hidden_dim, 1))

        # long short term memory
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=False)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, g, nfeats):
        """Forward function"""
        names = g.ndata["names"]
        centrality = g.ndata["centrality"]

        with g.local_scope():

            nfeats = torch.cat((nfeats, names), dim=1) * centrality

            hidden_global0 = self.global_attention0(g, nfeats)
            # Apply graph convolution and activation.
            hidden_states = F.relu(self.conv1(g, nfeats, edge_weight=g.edata["weights"]))
            hidden_states = torch.mean(hidden_states, dim=1)
            hidden_global1 = self.global_attention1(g, hidden_states)
            hidden_states = F.relu(self.conv2(g, hidden_states, edge_weight=g.edata["weights"]))
            hidden_states = torch.mean(hidden_states, dim=1)
            hidden_global2 = self.global_attention2(g, hidden_states)
            hidden_states = F.relu(self.conv3(g, hidden_states, edge_weight=g.edata["weights"]))
            hidden_states = torch.mean(hidden_states, dim=1)
            hidden_global3 = self.global_attention3(g, hidden_states)

            hidden_global = torch.stack((hidden_global0, hidden_global1, hidden_global2, hidden_global3), dim=0)

            hidden_global, (last_hidden_lstm, _) = self.lstm(hidden_global)
            hidden_global = F.dropout(last_hidden_lstm[-1, :, :], p=self.dropout, training=self.training)
            return F.sigmoid(self.classifier(hidden_global))


# GRAPHORMER MODELS
class SmallGraphormer(nn.Module):
    """Class for 6-layered Graphormer"""
    def __init__(self, num_features, transformer_hidden_dim, dropout, attn_dropout, num_heads):
        super().__init__()
        self.num_features = num_features

        # centrality encoder for the degree
        self.centrality_encoder = dglnn.DegreeEncoder(max_degree=100, embedding_dim=num_features)

        # spacial encoder for attention bias
        self.encoder = dglnn.SpatialEncoder(max_dist=10, num_heads=num_heads)

        # transformer to update embeddings with attention
        self.transformer1 = dglnn.GraphormerLayer(feat_size=num_features, hidden_size=transformer_hidden_dim, num_heads=num_heads, dropout=dropout, attn_dropout=attn_dropout)
        self.transformer2 = dglnn.GraphormerLayer(feat_size=num_features, hidden_size=transformer_hidden_dim, num_heads=num_heads, dropout=dropout, attn_dropout=attn_dropout)
        self.transformer3 = dglnn.GraphormerLayer(feat_size=num_features, hidden_size=transformer_hidden_dim, num_heads=num_heads, dropout=dropout, attn_dropout=attn_dropout)
        self.transformer4 = dglnn.GraphormerLayer(feat_size=num_features, hidden_size=transformer_hidden_dim, num_heads=num_heads, dropout=dropout, attn_dropout=attn_dropout)
        self.transformer5 = dglnn.GraphormerLayer(feat_size=num_features, hidden_size=transformer_hidden_dim, num_heads=num_heads, dropout=dropout, attn_dropout=attn_dropout)
        self.transformer6 = dglnn.GraphormerLayer(feat_size=num_features, hidden_size=transformer_hidden_dim, num_heads=num_heads, dropout=dropout, attn_dropout=attn_dropout)

        # classifier to predict the class
        self.classifier = nn.Linear(num_features, 1)

    def forward(self, g, nfeats):
        """Forward function"""
        g_list = dgl.unbatch(g)
        max_num_nodes = torch.max(g.batch_num_nodes())
        features = torch.zeros(len(g_list), max_num_nodes.item(), self.num_features).to(DEVICE)
        attn_mask = torch.zeros(len(g_list), max_num_nodes.item(), max_num_nodes.item()).to(DEVICE)

        batch_n = 0
        accum_nodes = 0
        for graph in g_list:
            features[batch_n, :graph.number_of_nodes()] = nfeats[accum_nodes:accum_nodes+graph.number_of_nodes()] + self.centrality_encoder(graph)
            attn_mask[batch_n, :, graph.number_of_nodes():] = 1
            attn_mask[batch_n, graph.number_of_nodes():, 1:] = 1
            batch_n += 1
            accum_nodes += graph.number_of_nodes()

        # compute spatial encoding
        bias = self.encoder(g)

        # apply transformer
        hidden_states = self.transformer1(features, bias, attn_mask=attn_mask)
        hidden_states = self.transformer2(hidden_states, bias, attn_mask=attn_mask)
        hidden_states = self.transformer3(hidden_states, bias, attn_mask=attn_mask)
        hidden_states = self.transformer4(hidden_states, bias, attn_mask=attn_mask)
        hidden_states = self.transformer5(hidden_states, bias, attn_mask=attn_mask)
        hidden_states = self.transformer6(hidden_states, bias, attn_mask=attn_mask)

        hidden_global = torch.zeros(len(g_list), self.num_features).to(DEVICE)
        batch_n = 0
        for graph in g_list:
            hidden_global[batch_n] = torch.mean(hidden_states[batch_n, :graph.number_of_nodes(), :], dim=0)
            batch_n += 1
        hidden_global = F.dropout(hidden_global, p=0.2, training=self.training)
        return F.sigmoid(self.classifier(hidden_global))
