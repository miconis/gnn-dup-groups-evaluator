"""Module providing datasets api"""

import string
import unicodedata
import json
import os
import jaro
from networkx.algorithms.centrality import betweenness_centrality
import numpy as np
from dgl.data import DGLDataset
import torch
from dgl.data.utils import save_graphs, load_graphs
import dgl
# from pyspark import SparkConf, SparkContext
# from pyspark.sql import SparkSession


ALL_LETTERS = string.ascii_letters + ".,-"

def is_correct(x):
    """
    Check if a group is correct (based on the orcids)
    """
    orcids = []
    for doc in x['docs']:
        orcids.append(json.loads(doc)['orcid'])
    return len(set(orcids)) == 1


def has_unique_authors(group):
    """
    Check if all the authors in the group have different ids
    """
    ids = [json.loads(doc)['id'] for doc in group['docs']]
    return len(ids) == len(set(ids))


def is_group(group):
    """
    Check if the group is actually a graph (more than 2 nodes)
    """
    return len(group['docs']) > 2


def unicode_to_ascii(s):
    """
    return ascii from unicode
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )


def letter_index(letter):
    """
    return letter index
    """
    return ALL_LETTERS.find(letter)


def letter_to_tensor(letter):
    """
    return tensor from letter
    """
    tensor = torch.zeros(1, len(ALL_LETTERS))
    tensor[0][letter_index(letter)] = 1
    return tensor


def name_to_tensor(name):
    """
    return name from tensor
    """
    tensor = torch.zeros(len(ALL_LETTERS), len(name))
    for idx, letter in enumerate(name):
        tensor[letter_index(letter)][idx] = 1
    return tensor

class DedupGroupsDataset(DGLDataset):
    """
    Dataset for Graph Classification of Dedup Groups.

    Parameters
    ----------
    url : URL to download the raw dataset
    raw_dir : directory that will store (or already stores) the downloaded data
    save_dir : directory to save preprocessed data
    force_reload : whether to reload dataset
    verbose : whether to print out progress information
    """
    def __init__(self,
                 dataset_name,
                 dedup_groups_path,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):

        # self.conf = SparkConf() \
        #             .setAppName("Dataset Processor") \
        #             .set("spark.driver.memory", "15g") \
        #             .setMaster("local[*]")
        # self.sc = SparkContext(conf=self.conf)
        # self.spark = SparkSession.builder.config(conf=self.conf).getOrCreate()
        self.dedup_groups_path = dedup_groups_path
        self.graphs = []
        self.labels = []
        super().__init__(name=dataset_name,
                                                 url=url,
                                                 raw_dir=raw_dir,
                                                 save_dir=save_dir,
                                                 force_reload=force_reload,
                                                 verbose=verbose)

    def download(self):
        """
        Download the raw data to local disk
        """
        print("Downloading data!")
        # groups = self.sc.textFile(self.dedup_groups_path).map(json.loads)
        # # filter out groups with duplicated authors (same entity with same id)
        # groups = groups.filter(has_unique_authors)
        # # filter out groups with only 2 authors (they are not groups!)
        # groups = groups.filter(is_group)
        #
        # # make the dataset balanced
        # correct_groups = groups.filter(is_correct)
        # wrong_groups = groups.filter(lambda x: not is_correct(x))
        #
        # # calculate the cardinality of the class with less elements
        # size = min(correct_groups.count(), wrong_groups.count())
        #
        # # take size element from each class
        # correct_groups = correct_groups.zipWithIndex().filter(lambda x: x[1] < size).map(lambda x: x[0])
        # wrong_groups = wrong_groups.zipWithIndex().filter(lambda x: x[1] < size).map(lambda x: x[0])
        #
        # # union of the classes
        # groups = correct_groups.union(wrong_groups).map(json.dumps)
        #
        # groups.coalesce(1).saveAsTextFile(self.raw_dir + "/" + self.name)
        # sc.stop()
        # print("Data dowloaded!")

    def process(self):
        """
        Process raw data to graphs and labels
        """
        with open(self.raw_dir + "/" + self.name + "/part-00000") as groups:
            for group in groups:
                try:
                    json_group = json.loads(group)
                    graph, label = self.dedup_group_to_graph(json_group)
                    self.graphs.append(graph)
                    self.labels.append(label)
                except Exception as exception:
                    print(exception)
                    print(group)
                    exit()

    def dedup_group_to_graph(self, x):
        """
        return DGLGraphs from groups
        """
        orcids = [json.loads(doc)['orcid'] for doc in x['docs']]
        ids = [json.loads(doc)['id'] for doc in x['docs']]
        label = 1 if len(set(orcids)) == 1 else 0

        node_feats = self.get_node_features_matrix(ids, x['docs'])
        edge_index = self.get_edges(ids, x['simrels'])

        names = list(map(unicode_to_ascii, [json.loads(doc)["fullname"] for doc in x["docs"]]))
        group_names = self.get_group_names_tensor(names)

        g = dgl.graph((edge_index[0], edge_index[1]))
        g = dgl.add_self_loop(g)

        centrality = self.get_node_centrality(g)

        edge_weights = self.get_edge_weights_tensor(names, g.edges())

        g.ndata["features"] = node_feats
        g.ndata["names"] = group_names
        g.ndata["centrality"] = centrality
        g.ndata["weights"] = edge_weights

        return g, label

    def get_node_centrality(self, g):
        """
        Compute the betweenness centrality of each node.
        """
        c = betweenness_centrality(dgl.to_networkx(g))
        return torch.tensor([c[key] for key in c])[:, None] + 1


    def get_group_names_tensor(self, names):
        """
        Produces the tensor of the names encoding
        """
        tensor = torch.zeros(len(names), len(ALL_LETTERS))
        for i, _ in enumerate(names):
            tensor[i, :] = name_to_tensor(names[i]).sum(dim=1)
        return tensor


    def get_edge_weights_tensor(self, names, edges):
        """
        Produces the weight for each edge
        """
        weights = []
        for i in range(len(edges[0])):
            edge = (edges[0][i], edges[1][i])
            name1 = names[edge[0]]
            name2 = names[edge[1]]
            weights.append(jaro.jaro_winkler_metric(name1, name2))
        return torch.tensor(weights)


    def get_node_features_matrix(self, _, docs):
        """
        Produces the node features: an NxF matrix (N:number of nodes, F: number of features)
        """
        all_node_feats = []

        for node in docs:
            all_node_feats.append(json.loads(node)['embeddings']['bert_embedding'])

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def get_edges(self, ids, simrels):
        """
        Produces the adjacency list (a list of edges, bi-directional)
        """
        edge_indices = []

        for edge in simrels:
            try:
                i = ids.index(edge['source'])
                j = ids.index(edge['target'])
                edge_indices += [[i, j], [j, i]]
            except Exception:
                print("error in creating edge")

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def get_label(self):
        """return labels"""
        labels = np.asarray([self.labels])
        return {"glabel": torch.tensor(labels, dtype=torch.int64)}

    def __getitem__(self, idx):
        """
        Get one example by index
        """
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.graphs)

    def __num_abstract_features__(self):
        """
        Return the number of features extracted from the abstract
        """
        return 768

    def __num_names_features__(self):
        """
        Return the number of features extracted from the names in the author list
        """
        return len(ALL_LETTERS)

    def __num_classes__(self):
        """
        Return the number of classes
        """
        return 2

    def __stats__(self):
        """
        Return statistics on the dataset
        """
        counters = {"3positives": 0, "3negatives": 0, "4to10positives": 0, "4to10negatives": 0, "above10positives": 0, "above10negatives": 0}
        for i, _ in enumerate(self.graphs):
            if self.graphs[i].number_of_nodes() == 3:
                if self.labels[i] == 1:
                    counters["3positives"] += 1
                else:
                    counters["3negatives"] += 1
            if 4 <= self.graphs[i].number_of_nodes() <= 10:
                if self.labels[i] == 1:
                    counters["4to10positives"] += 1
                else:
                    counters["4to10negatives"] += 1
            if self.graphs[i].number_of_nodes() > 10:
                if self.labels[i] == 1:
                    counters["above10positives"] += 1
                else:
                    counters["above10negatives"] += 1
        return counters

    def save(self):
        """
        Save processed data to directory (self.save_path)
        """
        save_graphs(self.save_path, self.graphs, self.get_label())

    def load(self):
        """
        Load processed data from directory (self.save_path)
        """
        self.graphs, dict_labels = load_graphs(self.save_path)
        self.labels = dict_labels['glabel'][0].tolist()

    def has_cache(self):
        """
        Check whether there are processed data
        """
        return os.path.exists(self.save_path)



