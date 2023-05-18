import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"


# GRAPH CONVOLUTIONAL NETWORKS
class GCN3(nn.Module):
    def __init__(self, num_features, hidden_dim, dropout):
        super(GCN3, self).__init__()
        self.dropout = dropout
        self.conv1 = dglnn.GraphConv(num_features, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv3 = dglnn.GraphConv(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, g, nfeats):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, nfeats))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        with g.local_scope():
            g.ndata["h"] = h
            hg = dgl.mean_nodes(g, "h")
            hg = F.dropout(hg, p=self.dropout, training=self.training)
            return F.sigmoid(self.classifier(hg))


class GCN3WeightedEdges(nn.Module):
    def __init__(self, num_features, hidden_dim, dropout):
        super(GCN3WeightedEdges, self).__init__()
        self.dropout = dropout

        self.norm = dglnn.EdgeWeightNorm(norm="left")

        self.conv1 = dglnn.GraphConv(num_features, hidden_dim, norm="none")
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, norm="none")
        self.conv3 = dglnn.GraphConv(hidden_dim, hidden_dim, norm="none")

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, g, nfeats):

        norm_edge_weights = self.norm(g, g.edata["weights"])
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, nfeats, edge_weight=norm_edge_weights))
        h = F.relu(self.conv2(g, h, edge_weight=norm_edge_weights))
        h = F.relu(self.conv3(g, h, edge_weight=norm_edge_weights))
        with g.local_scope():
            g.ndata["h"] = h
            hg = dgl.mean_nodes(g, "h")
            hg = F.dropout(hg, p=self.dropout, training=self.training)
            return F.sigmoid(self.classifier(hg))


class GCN3Names(nn.Module):
    def __init__(self, num_features, hidden_dim, dropout):
        super(GCN3Names, self).__init__()
        self.dropout = dropout
        self.conv1 = dglnn.GraphConv(num_features, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv3 = dglnn.GraphConv(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, g, nfeats):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, torch.cat((nfeats, g.ndata["names"]), dim=1)))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        with g.local_scope():
            g.ndata["h"] = h
            hg = dgl.mean_nodes(g, "h")
            hg = F.dropout(hg, p=self.dropout, training=self.training)
            return F.sigmoid(self.classifier(hg))


# GRAPH ATTENTION NETWORKS
class GAT3(nn.Module):
    def __init__(self, num_features, hidden_dim, num_heads, dropout, feat_drop, attn_drop):
        super(GAT3, self).__init__()
        self.dropout = dropout
        self.conv1 = dglnn.GATConv(num_features, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)
        self.conv2 = dglnn.GATConv(hidden_dim, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)
        self.conv3 = dglnn.GATConv(hidden_dim, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, g, nfeats):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, nfeats))
        h = torch.mean(h, dim=1)
        h = F.relu(self.conv2(g, h))
        h = torch.mean(h, dim=1)
        h = F.relu(self.conv3(g, h))
        h = torch.mean(h, dim=1)

        with g.local_scope():
            g.ndata["h"] = h
            hg = dgl.mean_nodes(g, "h")
            hg = F.dropout(hg, p=self.dropout, training=self.training)
            return F.sigmoid(self.classifier(hg))


class GAT3LSTM(nn.Module):
    def __init__(self, num_features, hidden_dim, num_heads, dropout, feat_drop, attn_drop, num_layers):
        super(GAT3LSTM, self).__init__()
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
        with g.local_scope():

            hg0 = self.global_attention0(g, nfeats)
            # Apply graph convolution and activation.
            h = F.relu(self.conv1(g, nfeats))
            h = torch.mean(h, dim=1)
            hg1 = self.global_attention1(g, h)
            h = F.relu(self.conv2(g, h))
            h = torch.mean(h, dim=1)
            hg2 = self.global_attention2(g, h)
            h = F.relu(self.conv3(g, h))
            h = torch.mean(h, dim=1)
            hg3 = self.global_attention3(g, h)

            hg = torch.stack((hg0, hg1, hg2, hg3), dim=0)

            hg, (hn, cn) = self.lstm(hg)
            hg = F.dropout(hn[-1, :, :], p=self.dropout, training=self.training)
            return F.sigmoid(self.classifier(hg))


class GAT3NamesLSTM(nn.Module):
    def __init__(self, num_features, hidden_dim, num_heads, dropout, feat_drop, attn_drop, num_layers):
        super(GAT3NamesLSTM, self).__init__()
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
        names = g.ndata["names"]
        with g.local_scope():

            hg0 = self.global_attention0(g, torch.cat((nfeats, names), dim=1))
            # Apply graph convolution and activation.
            h = F.relu(self.conv1(g, torch.cat((nfeats, names), dim=1)))
            h = torch.mean(h, dim=1)
            hg1 = self.global_attention1(g, h)
            h = F.relu(self.conv2(g, h))
            h = torch.mean(h, dim=1)
            hg2 = self.global_attention2(g, h)
            h = F.relu(self.conv3(g, h))
            h = torch.mean(h, dim=1)
            hg3 = self.global_attention3(g, h)

            hg = torch.stack((hg0, hg1, hg2, hg3), dim=0)

            hg, (hn, cn) = self.lstm(hg)
            hg = F.dropout(hn[-1, :, :], p=self.dropout, training=self.training)
            return F.sigmoid(self.classifier(hg))


class GAT3NamesEdgesLSTM(nn.Module):
    def __init__(self, num_features, hidden_dim, num_heads, dropout, feat_drop, attn_drop, num_layers):
        super(GAT3NamesEdgesLSTM, self).__init__()
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
        names = g.ndata["names"]
        with g.local_scope():

            hg0 = self.global_attention0(g, torch.cat((nfeats, names), dim=1))
            # Apply graph convolution and activation.
            h = F.relu(self.conv1(g, torch.cat((nfeats, names), dim=1), edge_weight=g.edata["weights"]))
            h = torch.mean(h, dim=1)
            hg1 = self.global_attention1(g, h)
            h = F.relu(self.conv2(g, h, edge_weight=g.edata["weights"]))
            h = torch.mean(h, dim=1)
            hg2 = self.global_attention2(g, h)
            h = F.relu(self.conv3(g, h, edge_weight=g.edata["weights"]))
            h = torch.mean(h, dim=1)
            hg3 = self.global_attention3(g, h)

            hg = torch.stack((hg0, hg1, hg2, hg3), dim=0)

            hg, (hn, cn) = self.lstm(hg)
            hg = F.dropout(hn[-1, :, :], p=self.dropout, training=self.training)
            return F.sigmoid(self.classifier(hg))


class GAT3NamesEdgesCentralityLSTM(nn.Module):
    def __init__(self, num_features, hidden_dim, num_heads, dropout, feat_drop, attn_drop, num_layers):
        super(GAT3NamesEdgesCentralityLSTM, self).__init__()
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
        names = g.ndata["names"]
        centrality = g.ndata["centrality"]

        with g.local_scope():

            nfeats = torch.cat((nfeats, names), dim=1) * centrality

            hg0 = self.global_attention0(g, nfeats)
            # Apply graph convolution and activation.
            h = F.relu(self.conv1(g, nfeats, edge_weight=g.edata["weights"]))
            h = torch.mean(h, dim=1)
            hg1 = self.global_attention1(g, h)
            h = F.relu(self.conv2(g, h, edge_weight=g.edata["weights"]))
            h = torch.mean(h, dim=1)
            hg2 = self.global_attention2(g, h)
            h = F.relu(self.conv3(g, h, edge_weight=g.edata["weights"]))
            h = torch.mean(h, dim=1)
            hg3 = self.global_attention3(g, h)

            hg = torch.stack((hg0, hg1, hg2, hg3), dim=0)

            hg, (hn, cn) = self.lstm(hg)
            hg = F.dropout(hn[-1, :, :], p=self.dropout, training=self.training)
            return F.sigmoid(self.classifier(hg))


# GRAPHORMER MODELS
class SmallGraphormer(nn.Module):
    def __init__(self, num_features, transformer_hidden_dim, dropout, attn_dropout, num_heads):
        super(SmallGraphormer, self).__init__()
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
        g_list = dgl.unbatch(g)
        max_num_nodes = torch.max(g.batch_num_nodes())
        features = torch.zeros(len(g_list), max_num_nodes.item(), self.num_features).to(device)
        attn_mask = torch.zeros(len(g_list), max_num_nodes.item(), max_num_nodes.item()).to(device)

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
        h = self.transformer1(features, bias, attn_mask=attn_mask)
        h = self.transformer2(h, bias, attn_mask=attn_mask)
        h = self.transformer3(h, bias, attn_mask=attn_mask)
        h = self.transformer4(h, bias, attn_mask=attn_mask)
        h = self.transformer5(h, bias, attn_mask=attn_mask)
        h = self.transformer6(h, bias, attn_mask=attn_mask)

        hg = torch.zeros(len(g_list), self.num_features).to(device)
        batch_n = 0
        for graph in g_list:
            hg[batch_n] = torch.mean(h[batch_n, :graph.number_of_nodes(), :], dim=0)
            batch_n += 1
        hg = F.dropout(hg, p=0.2, training=self.training)
        return F.sigmoid(self.classifier(hg))

