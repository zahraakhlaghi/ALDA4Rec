from Params import args
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
import numpy as np
from Utils.TimeLogger import log
from DataHandler import negSamp, negSamp_fre, transpose, DataHandler, transToLsts, transToLstsFloat
from Utils.attention import AdditiveAttention, MultiHeadSelfAttention
import scipy.sparse as sp
from random import randint
import networkx as nx
from community import community_louvain

from scipy.sparse import csr_matrix
from collections import Counter


class Recommender:
    def __init__(self, sess, handler):
        self.sess = sess
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'HR', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            init = tf.global_variables_initializer()
            self.sess.run(init)
            log('Variables Inited')
        maxndcg = 0.0
        maxres = dict()
        maxepoch = 0
        for ep in range(stloc, args.epoch):
            test = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, test))
            if test:
                reses = self.testEpoch()
                log(self.makePrint('Test', ep, reses, test))
            if ep % args.tstEpoch == 0 and reses['NDCG'] > maxndcg:
                self.saveHistory()
                maxndcg = reses['NDCG']
                maxres = reses
                maxepoch = ep
            print()
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        log(self.makePrint('max', maxepoch, maxres, True))

    # self.saveHistory()
    # def LightGcn(self, adj, )
    def makeTimeEmbed(self):
        divTerm = 1 / (10000 ** (tf.range(0, args.latdim * 2, 2, dtype=tf.float32) / args.latdim))
        pos = tf.expand_dims(tf.range(0, self.maxTime, dtype=tf.float32), axis=-1)
        sine = tf.expand_dims(tf.math.sin(pos * divTerm) / np.sqrt(args.latdim), axis=-1)
        cosine = tf.expand_dims(tf.math.cos(pos * divTerm) / np.sqrt(args.latdim), axis=-1)
        timeEmbed = tf.reshape(tf.concat([sine, cosine], axis=-1), [self.maxTime, args.latdim * 2]) / 4.0
        return timeEmbed

    def messagePropagate(self, srclats, mat, type='user'):
        timeEmbed = FC(self.timeEmbed, args.latdim, reg=True)
        srcNodes = tf.squeeze(tf.slice(mat.indices, [0, 1], [-1, 1]))
        tgtNodes = tf.squeeze(tf.slice(mat.indices, [0, 0], [-1, 1]))
        # print(srcNodes,tgtNodes)
        srcEmbeds = tf.nn.embedding_lookup(srclats, srcNodes)  # + tf.nn.embedding_lookup(timeEmbed, edgeVals)
        lat = tf.pad(tf.math.segment_sum(srcEmbeds, tgtNodes), [[0, 100], [0, 0]])
        if (type == 'user'):
            lat = tf.nn.embedding_lookup(lat, self.users)
        else:
            lat = tf.nn.embedding_lookup(lat, self.items)
        return Activate(lat, self.actFunc)


    def verify_adjacency_matrices(self):
        for k in range(args.graphNum):
            adj = self.subAdj[k].numpy()  # Convert to dense for verification
            max_user_index = adj.indices[:, 0].max()
            max_item_index = adj.indices[:, 1].max()
            assert max_user_index < args.user, f"Graph {k}: Max user index {max_user_index} exceeds args.user {args.user}"
            assert max_item_index < args.item, f"Graph {k}: Max item index {max_item_index} exceeds args.item {args.item}"

    def edgeDropout(self, mat):
        def dropOneMat(mat):
            # print("drop",mat)
            indices = mat.indices
            values = mat.values
            shape = mat.dense_shape
            # newVals = tf.to_float(tf.sign(tf.nn.dropout(values, self.keepRate)))
            newVals = tf.nn.dropout(tf.cast(values, dtype=tf.float32), self.keepRate)
            return tf.sparse.SparseTensor(indices, tf.cast(newVals, dtype=tf.int32), shape)

        return dropOneMat(mat)

    # cross-view collabrative Supervision
    def ours(self):
        user_vector, item_vector = list(), list()
        # user_vector_short,item_vector_short=list(),list()
        # embedding
        uEmbed = NNs.defineParam('uEmbed', [args.graphNum, args.user, args.latdim],
                                 reg=True)  # [graphNum, num_users, latent_dim]
        iEmbed = NNs.defineParam('iEmbed', [args.graphNum, args.item, args.latdim],
                                 reg=True)  # [graphNum, num_items, latent_dim]
        posEmbed = NNs.defineParam('posEmbed', [args.pos_length, args.latdim], reg=True)
        pos = tf.tile(tf.expand_dims(tf.range(args.pos_length), axis=0), [args.batch, 1])

        self.items = tf.range(args.item, dtype=tf.int32)
        self.users = tf.range(args.user, dtype=tf.int32)
        self.timeEmbed = NNs.defineParam('timeEmbed', [self.maxTime + 1, args.latdim], reg=True)

        for k in range(args.graphNum):
            embs0 = [uEmbed[k]]
            embs1 = [iEmbed[k]]
            for i in range(args.gnn_layer):
                a_emb0 = self.messagePropagate(embs1[-1], self.edgeDropout(self.subAdj[k]), 'user')
                a_emb1 = self.messagePropagate(embs0[-1], self.edgeDropout(self.subTpAdj[k]), 'item')
                embs0.append(a_emb0 + embs0[-1])
                embs1.append(a_emb1 + embs1[-1])
            user = tf.add_n(embs0)  # +tf.tile(timeUEmbed[k],[args.user,1])
            item = tf.add_n(embs1)  # +tf.tile(timeIEmbed[k],[args.item,1])
            user_vector.append(user)
            item_vector.append(item)
        # now user_vector is [g,u,latdim]

        user_vector = tf.stack(user_vector, axis=0)
        item_vector = tf.stack(item_vector, axis=0)
        user_vector_tensor = tf.transpose(user_vector, perm=[1, 0, 2])
        item_vector_tensor = tf.transpose(item_vector, perm=[1, 0, 2])
        #
        short_user_vector = tf.reduce_mean(user_vector_tensor, axis=1)  # +user_vector_long
        short_item_vector = tf.reduce_mean(item_vector_tensor, axis=1)  # +item_vector_long
        sUlat = tf.nn.embedding_lookup(short_user_vector, self.uids)
        sIlat = tf.nn.embedding_lookup(short_item_vector, self.iids)

        def gru_cell():
            return tf.contrib.rnn.BasicLSTMCell(args.latdim)

        def dropout():
            cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keepRate)

        with tf.name_scope("rnn"):
            cells = [dropout() for _ in range(1)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            user_vector_rnn, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=user_vector_tensor, dtype=tf.float32)
            item_vector_rnn, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=item_vector_tensor, dtype=tf.float32)
            user_vector_tensor = user_vector_rnn  # +user_vector_tensor
            item_vector_tensor = item_vector_rnn  # +item_vector_tensor
        self.additive_attention0 = AdditiveAttention(args.query_vector_dim, args.latdim)
        self.additive_attention1 = AdditiveAttention(args.query_vector_dim, args.latdim)

        self.multihead_self_attention0 = MultiHeadSelfAttention(args.latdim, args.num_attention_heads)
        self.multihead_self_attention1 = MultiHeadSelfAttention(args.latdim, args.num_attention_heads)
        multihead_user_vector = self.multihead_self_attention0.attention(tf.contrib.layers.layer_norm(
            user_vector_tensor))  # (tf.layers.batch_normalization(user_vector_tensor,training=self.is_train))#
        multihead_item_vector = self.multihead_self_attention1.attention(tf.contrib.layers.layer_norm(
            item_vector_tensor))  # (tf.layers.batch_normalization(item_vector_tensor,training=self.is_train))#
        final_user_vector = tf.reduce_mean(multihead_user_vector, axis=1)  # +user_vector_long
        final_item_vector = tf.reduce_mean(multihead_item_vector, axis=1)  # +item_vector_long
        iEmbed_att = final_item_vector
        # sequence att
        self.multihead_self_attention_sequence = list()
        for i in range(args.att_layer):
            self.multihead_self_attention_sequence.append(MultiHeadSelfAttention(args.latdim, args.num_attention_heads))
        sequence_batch = tf.contrib.layers.layer_norm(
            tf.matmul(tf.expand_dims(self.mask, axis=1), tf.nn.embedding_lookup(iEmbed_att, self.sequence)))
        sequence_batch += tf.contrib.layers.layer_norm(
            tf.matmul(tf.expand_dims(self.mask, axis=1), tf.nn.embedding_lookup(posEmbed, pos)))
        att_layer = sequence_batch
        for i in range(args.att_layer):
            att_layer1 = self.multihead_self_attention_sequence[i].attention(tf.contrib.layers.layer_norm(att_layer))
            att_layer = Activate(att_layer1, "leakyRelu") + att_layer
        att_user = tf.reduce_sum(att_layer, axis=1)
        # att_user=self.additive_attention0.attention(att_layer)# tf.reduce_sum(att_layer,axis=1)
        pckIlat_att = tf.nn.embedding_lookup(iEmbed_att, self.iids)
        pckUlat = tf.nn.embedding_lookup(final_user_vector, self.uids)
        pckIlat = tf.nn.embedding_lookup(final_item_vector, self.iids)

        meta11 = tf.concat([final_user_vector, short_user_vector], axis=-1)
        meta12 = FC(meta11, args.pSdim, useBias=True, activation='leakyRelu', reg=True, reuse=True, name="meta12")
        meta12 = FC(meta12, args.pSdim // 8, useBias=True, activation='leakyRelu', reg=True, reuse=True, name="meta14")
        user_weight = tf.squeeze(FC(meta12, 1, useBias=True, activation='sigmoid', reg=True, reuse=True, name="meta13"))
        pckSWeight = tf.nn.embedding_lookup(user_weight, self.uids)

        mean_preds = tf.reduce_sum(Activate(sUlat * sIlat, self.actFunc), axis=-1)
        long_preds = tf.reduce_sum(pckUlat * pckIlat, axis=-1) + tf.reduce_sum(
            Activate(tf.nn.embedding_lookup(att_user, self.uLocs_seq), "leakyRelu") * pckIlat_att,
            axis=-1)
        preds = (pckSWeight * long_preds) + ((1 - pckSWeight) * mean_preds)
        self.preds_one = list()
        self.final_one = list()
        for i in range(args.graphNum):
            pckUlat = tf.nn.embedding_lookup(user_vector[i], self.suids[i])
            pckIlat = tf.nn.embedding_lookup(item_vector[i], self.siids[i])
            preds_one = tf.reduce_sum(Activate(pckUlat * pckIlat, self.actFunc), axis=-1)
            self.preds_one.append(preds_one)

        return preds, mean_preds, long_preds, pckSWeight

    def export_sparse_tensor_to_coo(self, sparse_tensor, sess):
        """
        Converts a TensorFlow SparseTensor to a SciPy COO matrix.
        """
        indices, values, shape = sess.run([
            sparse_tensor.indices,
            sparse_tensor.values,
            sparse_tensor.dense_shape
        ])
        coo = sp.coo_matrix((values, (indices[:, 0], indices[:, 1])), shape=shape)
        return coo

    def compute_similarity(self, coo_matrix):
        """
        Computes the similarity matrix for items.
        """
        num_items = coo_matrix.shape[0]

        similarity_matrix = sp.lil_matrix((num_items, num_items))
        item_adj_squared = coo_matrix
        item_adj_squared = item_adj_squared.dot(item_adj_squared.T)

        # Ensure COO format and check for empty matrix
        if not isinstance(item_adj_squared, sp.coo_matrix):
            item_adj_squared = item_adj_squared.tocoo()
        if not item_adj_squared.nnz:
            print("Warning: Item adjacency matrix is empty. No similarities to compute.")
            return None

        row_sums = item_adj_squared.sum(axis=1).A1  # Convert to 1D array
        col_sums = item_adj_squared.sum(axis=0).A1  # Convert to 1D array

        # Iterate over non-zero elements
        rows, cols = item_adj_squared.nonzero()  # Only row and col are returned
        data = item_adj_squared.data  # Access the data values separately
        for i, j, value in zip(rows, cols, data):
            if i != j and (row_sums[i] + col_sums[j]) > 0:
                similarity = value / (row_sums[i] + col_sums[j] - 2)
                similarity_matrix[i, j] = similarity

        return similarity_matrix.tocsr()

    def find_similar_item_pairs(self, similarity_matrix):
        """
        Finds item pairs with similarity above the threshold.
        """
        coo = similarity_matrix.tocoo()
        val = self.max_similarity - self.min_similarity
        similar_pairs = {
            (row, col): data for row, col, data in zip(coo.row, coo.col, coo.data)
            if ((data-self.min_similarity)/val) > args.similarity_threshold and row != col
        }
        return similar_pairs

    def augment_similarity_graph(self, csr_matrix, csr_matrix_before, similar_pairs, item_similarity):

        num_users, num_items = csr_matrix.shape
        row, col = csr_matrix.nonzero()  # Only row and col are returned
        data = csr_matrix.data  # Access the data values separately

        matrix = item_similarity
        if matrix.shape[0] > 0:
            min_val = matrix.min()
            max_val = matrix.max()
            matrix_normalized = matrix
            if (max_val - min_val) > 0:
                matrix_normalized = (matrix - min_val) / (max_val - min_val)

        if not similar_pairs:
            print("Warning: No similar item pairs to augment the graph.")
            return csr_matrix  # Or return an empty CSR matrix

        # Track all existing edges in a set
        user_item_map = {(row[i], col[i]): csr_matrix[row[i], col[i]] for i in range(len(row))}
        existing_edges = set(zip(row, col))

        # Track new edges to avoid duplicates
        new_edges = []
        new_data = []

        # Iterate over similar pairs and add new edges
        for item1, item2 in similar_pairs.keys():
            users_with_item1 = set(csr_matrix[:, item1].nonzero()[0])

            # Add edges for users connected to item1
            for user in users_with_item1:
                if (user, item2) not in existing_edges:
                    new_edges.append((user, item2))
                    val = user_item_map.get((user, item1), None)
                    new_data.append(val * matrix_normalized[item1][item2])

        if len(new_edges) != len(new_data):
            print(f'{len(new_edges)} -=-=-=-= {len(new_data)}')

        if csr_matrix_before is not None:
            user_interactions = np.bincount(row, minlength=num_users)
            users_without_edge = np.where(user_interactions == 0)[0]
            for user in users_without_edge:
                start_idx = csr_matrix_before.indptr[user]
                end_idx = csr_matrix_before.indptr[user + 1]
                item_ids = csr_matrix_before.indices[start_idx:end_idx]
                values = csr_matrix_before.data[start_idx:end_idx]
                for val1, val2 in similar_pairs.keys():
                    if val1 in item_ids:
                        index = (item_ids == val1).nonzero()[0]
                        if index.size > 0:  # Ensure index is valid
                            new_edges.append((user, val2))
                            index = index[0]  # Only take the first index
                            new_data.append(values[index] * matrix_normalized[val1][val2])
                        else:
                            # Handle case where val1 is not found in item_ids
                            print(f"Warning: val1 {val1} not found in item_ids for user {user} {item_ids}")
                            # Optionally, append some default value or skip this entry
        # Prepare new edges
        if new_edges:
            new_rows, new_cols = zip(*new_edges)
        else:
            new_rows, new_cols = [], []
        if len(new_edges) != len(new_data):
            print(f'{len(new_edges)} ---- {len(new_data)}')

        # Combine existing and new edges
        augmented_csr = sp.csr_matrix((np.concatenate([data, new_data]),
                                       (np.concatenate([row, new_rows]),
                                        np.concatenate([col, new_cols]))),
                                      shape=(num_users, num_items))

        print("*******************")
        print(f'Number of edge in the original graph: {data.shape}, Augmented edge: {len(new_data)}')

        return augmented_csr

    def delete_noisy_edge(self, csr_matrix_, similarity, num_graph):
        similarity_matrix = similarity.dot(similarity.T) / 2
        num_users, num_items = csr_matrix_.shape
        row, col, data = [], [], []
        num_noise_nosed = 0
        for user in range(num_users):
            items = self.handler.user_item_sequence[num_graph][user]
            if len(items) < 3:
                row.extend([user] * len(items))
                col.extend(items)
                data.extend([1.] * len(items))
            else:
                submatrix = (similarity_matrix[items, :][:, items]).toarray()
                if submatrix.shape[0] > 0:
                    min_val = submatrix.min()
                    max_val = submatrix.max()
                    submatrix_normalized = submatrix
                    if (max_val - min_val) > 0:
                        submatrix_normalized = (submatrix - min_val) / (max_val - min_val)

                    G = nx.from_numpy_matrix(submatrix_normalized, create_using=nx.Graph)
                    partition = community_louvain.best_partition(G, weight='weight')
                    community_counts = Counter(partition.values())
                    noise_communities = [comm for comm, count in community_counts.items() if count < 2]
                    noise_nodes = [node for node, comm in partition.items() if comm in noise_communities]
                    num_noise_nosed += len(noise_nodes)

                    row.extend([user] * len(items))
                    col.extend(items)

                    data.extend([args.beta if n_ in noise_nodes else 1.0 for n_ in items])

            # Ensure lengths match after this block
            assert len(row) == len(col) == len(data), f"Length mismatch: {len(row)}, {len(col)}, {len(data)}"
        print(f'Num deleted nodes: {num_noise_nosed}')
        # After the loop, construct the CSR matrix
        return csr_matrix((data, (row, col)), shape=(num_users, num_items))

    def prepareModel(self):
        self.keepRate = tf.placeholder(dtype=tf.float32, shape=[])
        self.is_train = tf.placeholder_with_default(True, (), 'is_train')
        NNs.leaky = args.leaky
        self.actFunc = 'leakyRelu'
        adj = self.handler.trnMat
        idx, data, shape = transToLsts(adj, norm=True)
        self.adj = tf.sparse.SparseTensor(idx, data, shape)
        self.uids = tf.placeholder(name='uids', dtype=tf.int32, shape=[None])
        self.iids = tf.placeholder(name='iids', dtype=tf.int32, shape=[None])
        self.sequence = tf.placeholder(name='sequence', dtype=tf.int32, shape=[args.batch, args.pos_length])
        self.mask = tf.placeholder(name='mask', dtype=tf.float32, shape=[args.batch, args.pos_length])
        self.uLocs_seq = tf.placeholder(name='uLocs_seq', dtype=tf.int32, shape=[None])
        self.suids = list()
        self.siids = list()
        self.suLocs_seq = list()
        for k in range(args.graphNum):
            self.suids.append(tf.placeholder(name='suids%d' % k, dtype=tf.int32, shape=[None]))
            self.siids.append(tf.placeholder(name='siids%d' % k, dtype=tf.int32, shape=[None]))
            self.suLocs_seq.append(tf.placeholder(name='suLocs%d' % k, dtype=tf.int32, shape=[None]))
        self.subAdj = list()
        self.subTpAdj = list()
        self.subIAdj = list()
        # self.subAdjNp=list()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            robust_adj = []
            for i in range(args.graphNum):
                # generate item matrix
                itemadj = self.handler.itemMat[i]
                idx, data, shape = transToLstsFloat(itemadj, norm=False)
                print("3", shape)
                item_adj_sparse = tf.sparse.SparseTensor(idx, data, shape)
                self.subIAdj.append(item_adj_sparse)

                print("Exporting item adjacency matrix to COO format...")
                item_adj_coo = self.export_sparse_tensor_to_coo(item_adj_sparse, sess)
                print("Computing similarity...")
                # item_similarity_coo = self.compute_adamic_adar(item_adj_coo)
                item_similarity_coo = self.compute_similarity(item_adj_coo)
                print("similarity matrix computed.")
                print(item_similarity_coo)
                self.max_similarity = np.max(item_similarity_coo[item_similarity_coo.nonzero()])
                self.min_similarity = np.min(item_similarity_coo[item_similarity_coo.nonzero()])
                print(f'max:{self.max_similarity} min:{self.min_similarity}')

                # --- Threshold Similarities to Find Similar Item Pairs ---
                similar_item_pairs = self.find_similar_item_pairs(item_similarity_coo)
                print(f"Number of similar item pairs: {len(similar_item_pairs)}")

                # generate sequence matrix
                adj = self.handler.subMat[i]
                print(f"Delete less similar items")
                robust_adj.append(self.delete_noisy_edge(adj, item_similarity_coo, i))

                print(f"Augment similar items")
                if i == 0:
                    seqadj = self.augment_similarity_graph(robust_adj[i], None, similar_item_pairs,
                                                           item_similarity_coo.toarray())
                else:
                    seqadj = self.augment_similarity_graph(robust_adj[i], robust_adj[i - 1], similar_item_pairs,
                                                           item_similarity_coo.toarray())

                idx, data, shape = transToLstsFloat(seqadj, norm=True)
                print("1", shape)
                self.subAdj.append(tf.sparse.SparseTensor(idx, data, shape))
                idx, data, shape = transToLstsFloat(transpose(seqadj), norm=True)
                self.subTpAdj.append(tf.sparse.SparseTensor(idx, data, shape))
                print("2", shape)

                num_users = adj.shape[0]
                row, col = adj.nonzero()
                # row2, col2 = seqadj.nonzero()
                row2, col2 = seqadj.nonzero()

                item_interaction = np.bincount(col, minlength=adj.shape[1])
                user_interactions1 = np.bincount(row, minlength=num_users)  # Count occurrences of each user in 'row'
                user_interactions2 = np.bincount(row2, minlength=num_users)  # Count occurrences of each user in 'row'

                # Step 3: Print the results
                num_users_with_min = np.sum(user_interactions1 == np.min(user_interactions1))

                # Step 4: Print the results
                print(
                    f'max item interaction: {np.max(item_interaction)} min:{np.min(item_interaction)}\n '
                    f'num_min: {np.sum(item_interaction == np.min(item_interaction))} mean: {np.mean(item_interaction)} ')
                print(
                    f"Number of users with minimum interaction count: {num_users_with_min} ->{np.sum(user_interactions2 == np.min(user_interactions2))}")
                print(f"Min interactions: {np.min(user_interactions1)}->{np.min(user_interactions2)}")
                print(f"Max interactions: {np.max(user_interactions1)}->{np.max(user_interactions2)}")
                print(f"Mean interactions: {np.mean(user_interactions1)}->{np.mean(user_interactions2)}")

        self.maxTime = self.handler.maxTime
        #############################################################################
        self.preds, self.mean_preds, self.long_preds, self.pckSWeight = self.ours()
        sampNum = tf.shape(self.uids)[0] // 2
        self.posPred = tf.slice(self.preds, [0], [sampNum])  # begin at 0, size = sampleNum
        self.negPred = tf.slice(self.preds, [sampNum], [-1])  #

        self.meanPosPred = tf.slice(self.mean_preds, [0], [sampNum])  # begin at 0, size = sampleNum
        self.meanNegPred = tf.slice(self.mean_preds, [sampNum], [-1])

        self.longPosPred = tf.slice(self.long_preds, [0], [sampNum])  # begin at 0, size = sampleNum
        self.longNegPred = tf.slice(self.long_preds, [sampNum], [-1])

        self.posMeanWeight = tf.slice(self.pckSWeight, [0], [sampNum])
        self.negMeanWeight = tf.slice(self.pckSWeight, [sampNum], [-1])
        #
        self.meanPreLoss = tf.reduce_mean(tf.maximum(0.0, 1.0 - (
                ((self.posMeanWeight) * self.meanPosPred) - (self.negMeanWeight) * self.meanNegPred)))

        self.longPreLoss = tf.reduce_mean(
            tf.maximum(0.0, 1.0 - ((1 - self.posMeanWeight) * self.longPosPred - (
                    1 - self.negMeanWeight) * self.longNegPred)))  # +tf.reduce_mean(tf.maximum(0.0,self.negPred))

        self.preLoss = tf.reduce_mean(
            tf.maximum(0.0, 1.0 - (self.posPred - self.negPred)))  # +tf.reduce_mean(tf.maximum(0.0,self.negPred))

        self.regLoss = args.reg * Regularize()
        self.loss = (1 - args.lambda1) * self.meanPreLoss + args.lambda1 * self.longPreLoss + self.regLoss

        globalStep = tf.Variable(0, trainable=False)
        learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

    def sampleTrainBatch(self, batIds, labelMat, timeMat, train_sample_num):
        temTst = self.handler.tstInt[batIds]
        temLabel = labelMat[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * train_sample_num
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        uLocs_seq = [None] * temlen
        sequence = [None] * args.batch
        mask = [None] * args.batch
        cur = 0
        # utime = [[list(),list()] for x in range(args.graphNum)]
        for i in range(batch):
            posset = self.handler.sequence[batIds[i]][:-1]
            # posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
            sampNum = min(train_sample_num, len(posset))
            choose = 1
            if sampNum == 0:
                poslocs = [np.random.choice(args.item)]
                neglocs = [poslocs[0]]
            else:
                poslocs = []
                # choose = 1
                choose = randint(1, max(min(args.pred_num + 1, len(posset) - 3), 1))
                poslocs.extend([posset[-choose]] * sampNum)
                neglocs = negSamp(temLabel[i], sampNum, args.item, [self.handler.sequence[batIds[i]][-1], temTst[i]],
                                  self.handler.item_with_pop)
            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur + temlen // 2] = batIds[i]
                uLocs_seq[cur] = uLocs_seq[cur + temlen // 2] = i
                iLocs[cur] = posloc
                iLocs[cur + temlen // 2] = negloc
                cur += 1
            sequence[i] = np.zeros(args.pos_length, dtype=int)
            mask[i] = np.zeros(args.pos_length)
            posset = posset[:-choose]  # self.handler.sequence[batIds[i]][:-choose]
            if (len(posset) <= args.pos_length):
                sequence[i][-len(posset):] = posset
                mask[i][-len(posset):] = 1
            else:
                sequence[i] = posset[-args.pos_length:]
                mask[i] += 1
        uLocs = uLocs[:cur] + uLocs[temlen // 2: temlen // 2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen // 2: temlen // 2 + cur]
        uLocs_seq = uLocs_seq[:cur] + uLocs_seq[temlen // 2: temlen // 2 + cur]
        if (batch < args.batch):
            for i in range(batch, args.batch):
                sequence[i] = np.zeros(args.pos_length, dtype=int)
                mask[i] = np.zeros(args.pos_length)
        return uLocs, iLocs, sequence, mask, uLocs_seq  # ,utime

    def sampleSslBatch(self, batIds, labelMat, use_epsilon=True):
        temLabel = list()
        for k in range(args.graphNum):
            temLabel.append(labelMat[k][batIds].toarray())
        batch = len(batIds)
        temlen = batch * 2 * args.sslNum
        uLocs = [[None] * temlen] * args.graphNum
        iLocs = [[None] * temlen] * args.graphNum
        uLocs_seq = [[None] * temlen] * args.graphNum
        # epsilon=[[None] * temlen] * args.graphNum
        for k in range(args.graphNum):
            cur = 0
            for i in range(batch):
                posset = np.reshape(np.argwhere(temLabel[k][i] != 0), [-1])
                # print(posset)
                sslNum = min(args.sslNum, len(posset) // 2)  # len(posset)//4#
                if sslNum == 0:
                    poslocs = [np.random.choice(args.item)]
                    neglocs = [poslocs[0]]
                else:
                    all = np.random.choice(posset, sslNum * 2)  # - args.user
                    # print(all)
                    poslocs = all[:sslNum]
                    neglocs = all[sslNum:]
                for j in range(sslNum):
                    posloc = poslocs[j]
                    negloc = neglocs[j]
                    uLocs[k][cur] = uLocs[k][cur + 1] = batIds[i]
                    uLocs_seq[k][cur] = uLocs_seq[k][cur + 1] = i
                    iLocs[k][cur] = posloc
                    iLocs[k][cur + 1] = negloc
                    cur += 2
            uLocs[k] = uLocs[k][:cur]
            iLocs[k] = iLocs[k][:cur]
            uLocs_seq[k] = uLocs_seq[k][:cur]
        return uLocs, iLocs, uLocs_seq

    def trainEpoch(self):
        num = args.user
        sfIds = np.random.permutation(num)[:args.trnNum]
        epochLoss, epochPreLoss = [0] * 2
        num = len(sfIds)
        sample_num_list = [40]
        steps = int(np.ceil(num / args.batch))
        for s in range(len(sample_num_list)):
            for i in range(steps):
                st = i * args.batch
                ed = min((i + 1) * args.batch, num)
                batIds = sfIds[st: ed]

                target = [self.optimizer, self.meanPreLoss, self.longPreLoss, self.regLoss, self.loss, self.posPred,
                          self.negPred,
                          self.preds_one]
                feed_dict = {}
                uLocs, iLocs, sequence, mask, uLocs_seq = self.sampleTrainBatch(batIds, self.handler.trnMat,
                                                                                self.handler.timeMat,
                                                                                sample_num_list[s])
                # esuLocs, esiLocs, epsilon = self.sampleSslBatch(batIds, self.handler.subadj)
                suLocs, siLocs, suLocs_seq = self.sampleSslBatch(batIds, self.handler.subMat, False)
                feed_dict[self.uids] = uLocs
                feed_dict[self.iids] = iLocs
                # print("train",uLocs,uLocs_seq)
                feed_dict[self.sequence] = sequence
                feed_dict[self.mask] = mask
                feed_dict[self.is_train] = True
                feed_dict[self.uLocs_seq] = uLocs_seq

                for k in range(args.graphNum):
                    feed_dict[self.suids[k]] = suLocs[k]
                    feed_dict[self.siids[k]] = siLocs[k]
                    feed_dict[self.suLocs_seq[k]] = suLocs_seq[k]
                feed_dict[self.keepRate] = args.keepRate

                res = self.sess.run(target, feed_dict=feed_dict,
                                    options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

                meanPreLoss, longPreLoss, regLoss, loss, pos, neg, pone = res[1:]
                epochLoss += loss
                epochPreLoss += meanPreLoss + longPreLoss
                log('Step %d/%d: meanPreloss = %.2f, gruPreloss= %.2f, REGLoss = %.2f         ' % (
                    i + s * steps, steps * len(sample_num_list), meanPreLoss, longPreLoss, regLoss), save=False,
                    oneline=True)

        ret = dict()
        ret['Loss'] = epochLoss / steps
        ret['preLoss'] = epochPreLoss / steps
        return ret

    def sampleTestBatch(self, batIds, labelMat):  # labelMat=TrainMat(adj)
        batch = len(batIds)
        temTst = self.handler.tstInt[batIds]
        temLabel = labelMat[batIds].toarray()
        temlen = batch * args.testSize  # args.item
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        uLocs_seq = [None] * temlen
        tstLocs = [None] * batch
        sequence = [None] * args.batch
        mask = [None] * args.batch
        cur = 0
        val_list = [None] * args.batch
        for i in range(batch):
            if (args.test == True):
                posloc = temTst[i]
            else:
                posloc = self.handler.sequence[batIds[i]][-1]
                val_list[i] = posloc
            rdnNegSet = np.array(self.handler.test_dict[batIds[i] + 1][:args.testSize - 1]) - 1
            locset = np.concatenate((rdnNegSet, np.array([posloc])))
            tstLocs[i] = locset
            for j in range(len(locset)):
                uLocs[cur] = batIds[i]
                iLocs[cur] = locset[j]
                uLocs_seq[cur] = i
                cur += 1
            sequence[i] = np.zeros(args.pos_length, dtype=int)
            mask[i] = np.zeros(args.pos_length)
            if (args.test == True):
                posset = self.handler.sequence[batIds[i]]
            else:
                posset = self.handler.sequence[batIds[i]][:-1]
            # posset=self.handler.sequence[batIds[i]]
            if (len(posset) <= args.pos_length):
                sequence[i][-len(posset):] = posset
                mask[i][-len(posset):] = 1
            else:
                sequence[i] = posset[-args.pos_length:]
                mask[i] += 1
        if (batch < args.batch):
            for i in range(batch, args.batch):
                sequence[i] = np.zeros(args.pos_length, dtype=int)
                mask[i] = np.zeros(args.pos_length)
        return uLocs, iLocs, temTst, tstLocs, sequence, mask, uLocs_seq, val_list

    def testEpoch(self):
        epochHit, epochNdcg = [0] * 2
        epochHit5, epochNdcg5 = [0] * 2
        epochHit20, epochNdcg20 = [0] * 2
        epochHit1, epochNdcg1 = [0] * 2
        epochHit15, epochNdcg15 = [0] * 2
        ids = self.handler.tstUsrs
        num = len(ids)
        tstBat = args.batch
        steps = int(np.ceil(num / tstBat))
        # np.random.seed(100)
        for i in range(steps):
            st = i * tstBat
            ed = min((i + 1) * tstBat, num)
            batIds = ids[st: ed]
            feed_dict = {}
            uLocs, iLocs, temTst, tstLocs, sequence, mask, uLocs_seq, val_list = self.sampleTestBatch(batIds,
                                                                                                      self.handler.trnMat)
            suLocs, siLocs, _ = self.sampleSslBatch(batIds, self.handler.subMat, False)
            feed_dict[self.uids] = uLocs
            feed_dict[self.iids] = iLocs
            feed_dict[self.is_train] = False
            feed_dict[self.sequence] = sequence
            feed_dict[self.mask] = mask
            feed_dict[self.uLocs_seq] = uLocs_seq
            # print("test",uLocs_seq)
            for k in range(args.graphNum):
                feed_dict[self.suids[k]] = suLocs[k]
                feed_dict[self.siids[k]] = siLocs[k]
            feed_dict[self.keepRate] = 1.0
            preds = self.sess.run(self.preds, feed_dict=feed_dict,
                                  options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            if (args.uid != -1):
                print(preds[args.uid])
            if (args.test == True):
                hit, ndcg, hit5, ndcg5, hit20, ndcg20, hit1, ndcg1, hit15, ndcg15 = self.calcRes(
                    np.reshape(preds, [ed - st, args.testSize]), temTst, tstLocs)
            else:
                hit, ndcg, hit5, ndcg5, hit20, ndcg20, hit1, ndcg1, hit15, ndcg15 = self.calcRes(
                    np.reshape(preds, [ed - st, args.testSize]), val_list, tstLocs)
            epochHit += hit
            epochNdcg += ndcg
            epochHit5 += hit5
            epochNdcg5 += ndcg5
            epochHit20 += hit20
            epochNdcg20 += ndcg20
            epochHit15 += hit15
            epochNdcg15 += ndcg15
            epochHit1 += hit1
            epochNdcg1 += ndcg1
            log('Steps %d/%d: hit10 = %d, ndcg10 = %d' % (i, steps, hit, ndcg), save=False, oneline=True)
        ret = dict()
        ret['HR'] = epochHit / num
        ret['NDCG'] = epochNdcg / num
        print("epochNdcg1:{},epochHit1:{},epochNdcg5:{},epochHit5:{}".format(epochNdcg1 / num, epochHit1 / num,
                                                                             epochNdcg5 / num, epochHit5 / num))
        print("epochNdcg15:{},epochHit15:{},epochNdcg20:{},epochHit20:{}".format(epochNdcg15 / num, epochHit15 / num,
                                                                                 epochNdcg20 / num, epochHit20 / num))
        return ret

    def calcRes(self, preds, temTst, tstLocs):
        hit = 0
        ndcg = 0
        hit1 = 0
        ndcg1 = 0
        hit5 = 0
        ndcg5 = 0
        hit20 = 0
        ndcg20 = 0
        hit15 = 0
        ndcg15 = 0
        for j in range(preds.shape[0]):
            predvals = list(zip(preds[j], tstLocs[j]))
            predvals.sort(key=lambda x: x[0], reverse=True)
            shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
            if temTst[j] in shoot:
                hit += 1
                ndcg += np.reciprocal(np.log2(shoot.index(temTst[j]) + 2))
            shoot = list(map(lambda x: x[1], predvals[:5]))
            if temTst[j] in shoot:
                hit5 += 1
                ndcg5 += np.reciprocal(np.log2(shoot.index(temTst[j]) + 2))
            shoot = list(map(lambda x: x[1], predvals[:20]))
            if temTst[j] in shoot:
                hit20 += 1
                ndcg20 += np.reciprocal(np.log2(shoot.index(temTst[j]) + 2))
        return hit, ndcg, hit5, ndcg5, hit20, ndcg20, hit1, ndcg1, hit15, ndcg15

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        saver = tf.train.Saver()
        saver.save(self.sess, 'Models/' + args.save_path)
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, 'Models/' + args.load_model)
        with open('History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')
