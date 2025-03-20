import pickle
import numpy as np
from scipy.sparse import csr_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
from sklearn.preprocessing import normalize
import os
import random


def transpose(mat):
    coomat = sp.coo_matrix(mat)
    return csr_matrix(coomat.transpose())


def negSamp_fre(temLabel, sampSize, neg_frequency, pos_los):
    negset = [None] * sampSize
    cur = 0
    i = 0
    # print(temLabel)
    while cur < sampSize:
        rdmItm = neg_frequency[-i]  #
        # rdmItm = np.random.choice(args.item)
        # print(rdmItm,temLabel[rdmItm])
        if rdmItm != pos_los and temLabel[rdmItm] == 0:
            negset[cur] = rdmItm
            cur += 1
        i += 1
    return negset


def negSamp(temLabel, sampSize, nodeNum, trnPos, item_with_pop):
    negset = [None] * sampSize
    cur = 0
    # print(trnPos)
    while cur < sampSize:

        # rdmItm = random.choice(item_with_pop)
        # rdmItm = np.random.choice(sequence[rdmItm],1)
        rdmItm = np.random.choice(nodeNum)
        # if rdmItm not in temLabel and rdmItm != trnPos:
        if temLabel[rdmItm] == 0 and rdmItm not in trnPos:
            negset[cur] = rdmItm
            cur += 1
    return negset


def posSamp(user_sequence, sampleNum):
    indexs = np.random.choice(np.array(range(len(user_sequence))), sampleNum)
    # print(indexs)
    return user_sequence[indexs.sort()]


def transToLsts(mat, mask=False, norm=False):
    shape = [mat.shape[0], mat.shape[1]]
    coomat = sp.coo_matrix(mat)
    indices = np.array(list(map(list, zip(coomat.row, coomat.col))), dtype=np.int32)
    print(shape)
    data = coomat.data.astype(np.int32)

    if norm:
        rowD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=1) + 1e-8) + 1e-8)))
        colD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=0) + 1e-8) + 1e-8)))
        for i in range(len(data)):
            row = indices[i, 0]
            col = indices[i, 1]
            data[i] = data[i] * rowD[row] * colD[col]

    # half mask
    if mask:
        spMask = (np.random.uniform(size=data.shape) > 0.5) * 1.0
        data = data * spMask

    if indices.shape[0] == 0:
        indices = np.array([[0, 0]], dtype=np.int32)
        data = np.array([0.0], np.int32)
    return indices, data, shape


def transToLstsFloat(mat, mask=False, norm=False):
    shape = [mat.shape[0], mat.shape[1]]
    coomat = sp.coo_matrix(mat)  # Convert to COO sparse matrix
    indices = np.vstack((coomat.row, coomat.col)).transpose().astype(np.int32)
    data = coomat.data.astype(np.float32)

    if norm:
        # Symmetric normalization: D^{-1/2} * A * D^{-1/2}
        row_sum = np.array(coomat.sum(axis=1)).flatten()
        col_sum = np.array(coomat.sum(axis=0)).flatten()
        rowD = 1.0 / np.sqrt(row_sum + 1e-8)
        colD = 1.0 / np.sqrt(col_sum + 1e-8)

        data = data * rowD[coomat.row] * colD[coomat.col]  # Apply normalization

    # Half mask: randomly mask 50% of the data
    if mask:
        spMask = (np.random.uniform(size=data.shape) > 0.5).astype(np.float32)
        data = data * spMask

    # Handle empty matrix case
    if indices.shape[0] == 0:
        indices = np.array([[0, 0]], dtype=np.int32)
        data = np.array([0.0], dtype=np.float32)

    return indices, data, shape


class DataHandler:
    def __init__(self):
        if args.data == 'yelp':
            predir = './Datasets/Yelp/'
        elif args.data == 'gowalla':
            predir = './Datasets/gowalla/'
        elif args.data == 'amazon':
            predir = './Datasets/amazon/'
        else:
            predir = './Datasets/' + args.data + '/'
        self.predir = predir
        self.trnfile = predir + 'trn_mat_time'
        self.tstfile = predir + 'tst_int'
        self.sequencefile = predir + 'sequence'
        self.test_dictfile = predir + 'test_dict'

    def LoadData(self):
        if args.percent > 1e-8:
            print('noised')
            with open(self.predir + 'noise_%.2f' % args.percent, 'rb') as fs:
                trnMat = pickle.load(fs)
        else:
            with open(self.trnfile, 'rb') as fs:
                # print(pickle.load(fs))
                trnMat = pickle.load(fs)  # (pickle.load(fs) != 0).astype(np.float32)
        # test set
        with open(self.tstfile, 'rb') as fs:
            tstInt = np.array(pickle.load(fs))
        with open(self.sequencefile, 'rb') as fs:
            self.sequence = pickle.load(fs)
        if os.path.isfile(self.test_dictfile):
            with open(self.test_dictfile, 'rb') as fs:
                self.test_dict = pickle.load(fs)
        print("tstInt", tstInt)
        tstStat = (tstInt != None)
        print("tstStat", tstStat, len(tstStat))
        tstUsrs = np.reshape(np.argwhere(tstStat != False), [-1])
        print("tstUsrs", tstUsrs, len(tstUsrs))

        # self.trnMat = trnMat[0]
        def generate_rating_matrix_test(user_seq, num_users, num_items):
            # three lists are used to construct sparse matrix
            row = []
            col = []
            data = []
            for user_id, item_list in enumerate(user_seq):
                for item in item_list:  #
                    row.append(user_id)
                    col.append(item)
                    data.append(1)

            row = np.array(row)
            col = np.array(col)
            data = np.array(data)
            rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

            return rating_matrix

        def generate_item_matrix_test(user_seq, num_items):
            # Initialize sparse matrix directly
            item_matrix_time = []

            # Iterate over graphNum
            for g in range(args.graphNum):
                matrix = np.zeros((num_items, num_items))
                # Iterate through the user sequences
                for user_id, item_list in enumerate(user_seq[g]):
                    for i in range(1, len(item_list)):
                        num1 = item_list[i]
                        num2 = item_list[i - 1]
                        matrix[num1, num2] += 1
                        matrix[num2, num1] += 1

                # Normalize the matrix using sparse operations (L1 normalization)
                normalized_matrix = normalize(matrix, axis=1, norm='l1', copy=False)

                row, col, data = [], [], []
                print("***********************************")

                # Efficiently add non-zero entries
                for i in range(num_items):
                    for j in range(num_items):
                        if i == j:
                            row.append(i)
                            col.append(j)
                            data.append(1)
                        elif normalized_matrix[i, j] > args.item_threshold:
                            row.append(i)
                            col.append(j)
                            data.append(normalized_matrix[i, j])
                        # if i!=j and normalized_matrix[i, j] > args.item_threshold:
                        #    row.append(i)
                        #    col.append(j)
                        #    data.append(normalized_matrix[i, j])

                item_matrix_time.append(csr_matrix((data, (row, col)), shape=(num_items, num_items)))

            return item_matrix_time

        def get_user_items_ordered_by_time(intMat, usrnum):

            user_item_sequence = []

            for graph_idx, graph in enumerate(intMat):
                graph_coo = graph.tocoo()  # Convert CSR to COO for iteration
                user_item_order_graph = [[] for _ in range(usrnum)]
                for user, item, timestamp in zip(graph_coo.row, graph_coo.col, graph_coo.data):
                    user_item_order_graph[user].append((item, timestamp))

                # Sort the items for each user by time
                for user in range(usrnum):
                    user_item_order_graph[user].sort(key=lambda x: x[1])  # Sort by timestamp
                    user_item_order_graph[user] = [item for item, _ in user_item_order_graph[user]]  # Keep only items
                user_item_sequence.append(user_item_order_graph)

            return user_item_sequence

        args.user, args.item = trnMat[0].shape
        self.trnMat = generate_rating_matrix_test(self.sequence, args.user, args.item)
        self.user_item_sequence = get_user_items_ordered_by_time(trnMat[1], args.user)
        self.itemMat = generate_item_matrix_test(self.user_item_sequence, args.item)
        self.subMat = trnMat[1]
        self.timeMat = trnMat[2]
        print("trnMat", trnMat[0], trnMat[1], trnMat[2])
        self.tstInt = tstInt
        self.tstUsrs = tstUsrs
        self.prepareGlobalData()

    def timeProcess(self, trnMats):
        mi = 1e16
        ma = 0
        for i in range(len(trnMats)):
            minn = np.min(trnMats[i].data)
            maxx = np.max(trnMats[i].data)
            mi = min(mi, minn)
            ma = max(ma, maxx)
        maxTime = 0
        for i in range(len(trnMats)):
            newData = ((trnMats[i].data - mi) // (3600 * 24 * args.slot)).astype(np.int32)
            maxTime = max(np.max(newData), maxTime)
            trnMats[i] = csr_matrix((newData, trnMats[i].indices, trnMats[i].indptr), shape=trnMats[i].shape)
        print('MAX TIME', mi, ma, maxTime)
        return trnMats, maxTime + 1

    def prepareGlobalData(self):
        def tran_to_sym(R):
            adj_mat = sp.dok_matrix((args.user + args.item, args.user + args.item), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = R.tolil()
            adj_mat[:args.user, args.user:] = R
            adj_mat[args.user:, :args.user] = R.T
            adj_mat = adj_mat.tocsr()
            return (adj_mat + sp.eye(adj_mat.shape[0]))

        # adj = self.subMat
        self.maxTime = 1
        # self.subMat,self.maxTime=self.timeProcess(self.subMat)
        print(self.subMat[0], self.subMat[-1])

        self.item_with_pop = []

    def sampleLargeGraph(self, pckUsrs, pckItms=None, sampDepth=2, sampNum=args.graphSampleN, preSamp=False):
        adj = self.adj
        tpadj = self.tpadj

        def makeMask(nodes, size):
            mask = np.ones(size)
            if not nodes is None:
                mask[nodes] = 0.0
            return mask

        def updateBdgt(adj, nodes):
            if nodes is None:
                return 0
            tembat = 1000
            ret = 0
            for i in range(int(np.ceil(len(nodes) / tembat))):
                st = tembat * i
                ed = min((i + 1) * tembat, len(nodes))
                temNodes = nodes[st: ed]
                ret += np.sum(adj[temNodes], axis=0)
            return ret

        def sample(budget, mask, sampNum):
            score = (mask * np.reshape(np.array(budget), [-1])) ** 2
            norm = np.sum(score)
            if norm == 0:
                return np.random.choice(len(score), 1), sampNum - 1
            score = list(score / norm)
            arrScore = np.array(score)
            posNum = np.sum(np.array(score) != 0)
            if posNum < sampNum:
                pckNodes1 = np.squeeze(np.argwhere(arrScore != 0))
                # pckNodes2 = np.random.choice(np.squeeze(np.argwhere(arrScore==0.0)), min(len(score) - posNum, sampNum - posNum), replace=False)
                # pckNodes = np.concatenate([pckNodes1, pckNodes2], axis=0)
                pckNodes = pckNodes1
            else:
                pckNodes = np.random.choice(len(score), sampNum, p=score, replace=False)
            return pckNodes, max(sampNum - posNum, 0)

        def constructData(usrs, itms):
            adj = self.trnMat
            pckU = adj[usrs]
            tpPckI = transpose(pckU)[itms]
            pckTpAdj = tpPckI
            pckAdj = transpose(tpPckI)
            return pckAdj, pckTpAdj, usrs, itms

        usrMask = makeMask(pckUsrs, adj.shape[0])
        itmMask = makeMask(pckItms, adj.shape[1])
        itmBdgt = updateBdgt(adj, pckUsrs)
        if pckItms is None:
            pckItms, _ = sample(itmBdgt, itmMask, len(pckUsrs))
            itmMask = itmMask * makeMask(pckItms, adj.shape[1])
        usrBdgt = updateBdgt(tpadj, pckItms)
        uSampRes = 0
        iSampRes = 0
        for i in range(sampDepth + 1):
            uSamp = uSampRes + (sampNum if i < sampDepth else 0)
            iSamp = iSampRes + (sampNum if i < sampDepth else 0)
            newUsrs, uSampRes = sample(usrBdgt, usrMask, uSamp)
            usrMask = usrMask * makeMask(newUsrs, adj.shape[0])
            newItms, iSampRes = sample(itmBdgt, itmMask, iSamp)
            itmMask = itmMask * makeMask(newItms, adj.shape[1])
            if i == sampDepth or i == sampDepth and uSampRes == 0 and iSampRes == 0:
                break
            usrBdgt += updateBdgt(tpadj, newItms)
            itmBdgt += updateBdgt(adj, newUsrs)
        usrs = np.reshape(np.argwhere(usrMask == 0), [-1])
        itms = np.reshape(np.argwhere(itmMask == 0), [-1])
        return constructData(usrs, itms)