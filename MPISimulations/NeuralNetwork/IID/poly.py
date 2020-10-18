import time
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from scipy.special import logsumexp


def first_divisor(n):
    for i in range(2, n + 1):
        if n % i == 0:
            return i


def relu(x):
    return np.maximum(x, np.zeros_like(x))


def reul_deriv(x):
    tmp = np.copy(x)
    tmp[tmp != 0] = 1.
    return tmp


def loss(logits, target):
    a = -np.multiply(target, logits)
    a = np.sum(a, axis=1)
    b = logsumexp(logits, axis=1)
    return np.mean(a + b)


def softmax(logits):
    denom = logsumexp(logits, axis=1).reshape(-1, 1)
    return np.exp(logits - denom)


class Model:
    def __init__(self, dataset, batch_size, lr, h1, h2):
        if dataset == 'MNIST':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train, x_test = x_train.reshape(-1, 784) / 255., x_test.reshape(-1, 784) / 255.
            y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)
            self.num_classes = 10
            self.input_size = 784
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.lr = lr
        self.batch_size = batch_size
        self.h_1_size = h1
        self.h_2_size = h2
        self.W1 = 0.01 * np.random.randn(self.input_size, self.h_1_size)
        self.W2 = 0.01 * np.random.randn(self.h_1_size, self.h_2_size)
        self.W3 = 0.01 * np.random.randn(self.h_2_size, self.num_classes)
        self.iter_trained = 0
        self.train_acc = []
        self.test_acc = []
        self.batch_x = None
        self.batch_y = None
        self.y_2 = None
        self.y_3 = None
        self.y_4 = None
        self.error_2 = None
        self.error_3 = None
        self.error_4 = None
        self.delta_W_1 = None
        self.delta_W_2 = None
        self.delta_W_3 = None

    def get_random_batch(self):
        idx = np.random.permutation(len(self.x_train))[0:self.batch_size]
        batch_x, batch_y = self.x_train[idx], self.y_train[idx]
        # batch_x = np.concatenate([batch_x, np.ones([self.batch_size, 1])], axis=1)
        return batch_x, batch_y

    def get_train_job(self, piece_idx):
        if piece_idx == 0:
            self.batch_x, self.batch_y = self.get_random_batch()
            return self.batch_x, self.W1
        elif piece_idx == 1:
            return self.y_2, self.W2
        elif piece_idx == 2:
            return self.y_3, self.W3
        elif piece_idx == 3:
            return np.transpose(self.y_3), np.multiply(self.error_4, reul_deriv(self.y_4))
        elif piece_idx == 4:
            return np.multiply(self.error_4, reul_deriv(self.y_4)), self.W3.T
        elif piece_idx == 5:
            return self.y_2.T, np.multiply(self.error_3, reul_deriv(self.y_3))
        elif piece_idx == 6:
            return np.multiply(self.error_3, reul_deriv(self.y_3)), self.W2.T
        elif piece_idx == 7:
            return self.batch_x.T, np.multiply(self.error_2, reul_deriv(self.y_2))

    def push_train_job(self, piece_idx, res):
        if piece_idx == 0:
            self.y_2 = relu(res)
        elif piece_idx == 1:
            self.y_3 = relu(res)
        elif piece_idx == 2:
            self.y_4 = res
            # # print(self.loss[-1])
            self.error_4 = (-self.batch_y + softmax(self.y_4)) / self.batch_size
        elif piece_idx == 3:
            self.delta_W_3 = res
        elif piece_idx == 4:
            self.error_3 = res
        elif piece_idx == 5:
            self.delta_W_2 = res
        elif piece_idx == 6:
            self.error_2 = res
        elif piece_idx == 7:
            self.delta_W_1 = res
            if np.linalg.norm(self.delta_W_1, 2) < 20 and np.linalg.norm(self.delta_W_2, 2) < 20 and np.linalg.norm(self.delta_W_3, 2) < 20:
                self.W1 -= self.lr * self.delta_W_1
                self.W2 -= self.lr * self.delta_W_2
                self.W3 -= self.lr * self.delta_W_3
            else:
                print('Update skipped due to numerical issues')
            train_acc, _, test_acc, _ = self.performance()
            self.train_acc.append(train_acc)
            self.test_acc.append(test_acc)
            self.iter_trained += 1

    def performance(self):
        self.batch_x, self.batch_y = self.get_random_batch()
        y_2 = relu(self.batch_x @ self.W1)
        y_3 = relu(y_2 @ self.W2)
        y_4 = y_3 @ self.W3
        train_loss = loss(y_4, self.batch_y)
        t = np.argmax(self.batch_y, axis=1)
        pred = np.argmax(y_4, axis=1)
        train_acc = np.sum(t == pred) / len(t) * 100
        y_2_test = relu(self.x_test[0:self.batch_size] @ self.W1)
        y_3_test = relu(y_2_test @ self.W2)
        y_4_test = y_3_test @ self.W3
        test_loss = loss(y_4_test, self.y_test[0:self.batch_size])
        t = np.argmax(self.y_test[0:self.batch_size], axis=1)
        pred = np.argmax(y_4_test, axis=1)
        test_acc = np.sum(t == pred) / len(t) * 100
        return train_acc, train_loss, test_acc, test_loss


def poly_encode(A, B, kA, kB, n):
    k = kA * kB
    [t, r] = np.shape(A)
    [t, w] = np.shape(B)
    node_points = -1 + 2 * (np.array(list(range(n))) + 1) / n  ## RS nodes are uniformly spaced in [-1,1]
    workers = np.array(list(range(n)))

    Coding_matrix1 = np.zeros((n, k), dtype=float)
    for j in range(0, n):
        Coding_matrix1[j, :] = (node_points[j]) ** np.array(list(range(k)))
    Coding_A = Coding_matrix1[:, ::kB]
    c = int(r / kA)
    W1a = {}
    for i in range(0, kA):
        W1a[i] = A[:, i * c:(i + 1) * c]
    W2a = {}
    (uu, vv) = np.shape(W1a[0])
    for i in range(0, n):
        W2a[i] = np.zeros((uu, vv), dtype=float)
        for j in range(0, kA):
            W2a[i] = W2a[i] + Coding_A[i, j] * W1a[j]
    Coding_B = Coding_matrix1[:, 0:kB]
    d = int(w / kB)
    W1b = {}
    for i in range(0, kB):
        W1b[i] = B[:, i * d:(i + 1) * d]
    W2b = {}
    (uu, vv) = np.shape(W1b[0])
    for i in range(0, n):
        W2b[i] = np.zeros((uu, vv), dtype=float)
        for j in range(0, kB):
            W2b[i] = W2b[i] + Coding_B[i, j] * W1b[j]
    return W2a, W2b


def poly_decode(A_size, B_size, kA, kB, worker_product, node_idx, n):
    [t, r] = A_size
    [t, w] = B_size
    r_p = r // kA
    c_p = w // kB
    node_points = -1 + 2 * (np.array(list(range(n))) + 1) / n
    nodes = node_points[node_idx]
    reduced_products = worker_product
    c = r_p
    d = c_p
    k = kA * kB
    Coding_matrix = np.zeros((k, k), dtype=float)
    for j in range(0, k):
        Coding_matrix[j, :] = (nodes[j]) ** np.array(list(range(k)))
    decoding_mat = np.linalg.inv(Coding_matrix)
    (g, h) = (r_p, c_p)
    CC = [reduced_products[i].reshape(-1) for i in range(len(reduced_products))]
    if len(CC) == 1:
        CC = CC[0]
    else:
        CC = np.concatenate(CC, axis=0)
    BB = np.zeros((c * d, k), dtype=float)
    for i in range(0, k):
        BB[:, i] = CC[i * c * d:(i + 1) * c * d]
    decoded_blocks = np.matmul(BB, np.transpose(decoding_mat))
    final_res = np.zeros([r, w])
    for idx in range(kA * kB):
        row = idx // kB
        col = idx % kB
        to_go = np.reshape(decoded_blocks[:, idx], (g, h))
        final_res[row * r_p:row * r_p + r_p, col * c_p:col * c_p + c_p] = to_go
    return final_res



def master():
    ### Initialization Phase
    # Creating Straggling map
    processing_times = np.zeros([num_workers, num_epochs*5*8+1000])
    slot = 0
    encoding_times = []
    decoding_times = []
    load_per_round = []
    if channel_type == 'iid':
        straggling_map = np.random.binomial(1, erasure_prob, size=[num_workers, approx_times])
    elif channel_type == 'markov':
        straggling_map = np.zeros([num_workers, approx_times])
        for slot_idx in range(1, approx_times):
            for worker_idx in range(num_workers):
                if straggling_map[worker_idx, slot_idx - 1] == 0:
                    straggling_map[worker_idx, slot_idx] = np.random.binomial(1, phs)
                else:
                    straggling_map[worker_idx, slot_idx] = np.random.binomial(1, 1 - psh)
    # Sending straggling map to users
    str_map_req = []
    for worker_idx in range(num_workers):
        str_map_req.append(comm.Isend(straggling_map, dest=worker_idx + 1, tag=1000))
    MPI.Request.waitall(str_map_req)
    # Creating Models
    # Creating Models

    model1 = Model('MNIST', 1024, 0.1, h1, h2)
    model2 = Model('MNIST', 1024, 0.15, h1, h2)
    model3 = Model('MNIST', 1024, 0.2, h1, h2)
    model4 = Model('MNIST', 1024, 0.25, h1, h2)
    model5 = Model('MNIST', 1024, 0.3, h1, h2)
    models = [model1, model2, model3, model4, model5]
    model_under_operation = 0
    crt_job_per_model = np.zeros(len(models))
    decode = 0
    while 1:
        # determine whether there is anything to do
        crt_iteration = models[-1].iter_trained
        if model_under_operation == len(models) - 1 and crt_job_per_model[model_under_operation] == 0:
            print(crt_iteration)
        if crt_iteration == num_epochs:
            status = 'done'
        else:
            status = 'work'
        for worker_idx in range(num_workers):
            comm.send(status, dest=worker_idx + 1, tag=0)
        if status == 'done':
            break
        # extract the job and divide
        init_enc = time.time()
        A, B = models[model_under_operation].get_train_job(crt_job_per_model[model_under_operation])

        # pad zeros to fix any size problem
        A_shape_orig, B_shape_orig = np.shape(A), np.shape(B)

        new_A, new_B = A, B
        if A_shape_orig[0] % kA != 0:
            new_A = np.zeros([A_shape_orig[0] + kA - (A_shape_orig[0] % kA), A_shape_orig[1]])
            new_A[0:A_shape_orig[0], :] = A
        if B_shape_orig[1] % kB != 0:
            new_B = np.zeros([B_shape_orig[0], B_shape_orig[1] + kB - (B_shape_orig[1] % kB)])
            new_B[:, 0:B_shape_orig[1]] = B
        coded_A, coded_B = poly_encode(new_A.T, new_B, kA, kB, num_workers)
        encoding_times.append(time.time() - init_enc)
        size_req = []
        for worker_idx in range(num_workers):
            size_req.append(
                comm.Isend(np.array([coded_A[worker_idx].shape, coded_B[worker_idx].shape], dtype=np.int)
                           , dest=worker_idx + 1, tag=0))
        MPI.Request.waitall(size_req)
        req_A = []
        req_B = []
        for worker_idx in range(num_workers):
            shapeA = np.shape(coded_A[worker_idx])
            shapeB = np.shape(coded_B[worker_idx])
            load = shapeA[0] * shapeA[1] * shapeB[1] * len(coded_A)
            req_A.append(comm.Isend(np.ascontiguousarray(coded_A[worker_idx]), dest=worker_idx + 1, tag=0))
            req_B.append(comm.Isend(np.ascontiguousarray(coded_B[worker_idx]), dest=worker_idx + 1, tag=1))
        load_per_round.append(load)
        MPI.Request.waitall(req_A)
        MPI.Request.waitall(req_B)
        if decode:
            init_dec = time.time()
            recon = poly_decode(padded_A_size_to_decode, padded_B_size_to_decode, kA, kB
                                , [result_to_decode[i] for i in rec_idx_to_decode]
                                , rec_idx_to_decode,
                                num_workers)
            models[modeol_idx_to_deocde].push_train_job(job_idx_to_decode, recon[0:orig_A_size_to_decode[0], 0:orig_B_size_to_decode[1]])
            decoding_times.append(time.time() - init_dec)

        req_result = []
        result = [np.empty([int(new_A.shape[0] / kA), int(new_B.shape[1] / kB)]) for _ in range(num_workers)]
        for worker_idx in range(num_workers):
            req_result.append(comm.Irecv(result[worker_idx], source=worker_idx+1, tag=10))
        MPI.Request.waitall(req_result)
        for worker_idx in range(num_workers):
            processing_times[worker_idx, slot] = comm.recv(source=worker_idx+1, tag=0)
        slot += 1
        decode = 1
        result_to_decode = result
        rec_idx_to_decode = np.argsort(processing_times[:, slot])[0:recov_threshold]
        padded_A_size_to_decode, padded_B_size_to_decode = (new_A.T).shape, new_B.shape
        orig_A_size_to_decode, orig_B_size_to_decode = A.shape, B.shape
        modeol_idx_to_deocde = model_under_operation
        job_idx_to_decode = crt_job_per_model[model_under_operation]
        crt_job_per_model[model_under_operation] += 1
        crt_job_per_model[model_under_operation] = crt_job_per_model[model_under_operation] % 8
        model_under_operation += 1
        model_under_operation = model_under_operation % len(models)

    result = [] # train_acc, train_loss, test_acc, test_loss
    for mdl in models:
        result.append(mdl.performance())
    print(encoding_times[0:10])
    print(decoding_times[0:10])
    print(processing_times[0:10])
    train_acc = []
    test_acc = []
    for i in range(5):
        train_acc.append(models[i].train_acc)
        test_acc.append(models[i].test_acc)
    np.savetxt(
        'train_acc_per_iteration_poly_' + str(erasure_prob) + '_h1_' + str(h1) + '_h2_' + str(h2) + '_epochs_' + str(
            num_epochs) + '.txt',
        np.array(train_acc))
    np.savetxt(
        'test_acc_per_iteration_poly_' + str(erasure_prob) + '_h1_' + str(h1) + '_h2_' + str(h2) + '_epochs_' + str(
            num_epochs) + '.txt',
        np.array(test_acc))
    np.savetxt('result_poly_'+str(erasure_prob)+'_h1_'+str(h1)+'_h2_'+str(h2)+'_epochs_'+str(num_epochs),
               np.array(result))
    np.savetxt('encoding_time_poly_p_'+str(erasure_prob)+'_h1_'+str(h1)+'_h2_'+str(h2)+'_epochs_'+str(num_epochs),
               np.array(encoding_times))
    np.savetxt('decoding_time_poly_p_' + str(erasure_prob) + '_h1_' + str(h1) + '_h2_' + str(h2) + '_epochs_' + str(
        num_epochs),
               np.array(decoding_times))
    np.savetxt('processing_time_poly_p_' + str(erasure_prob) + '_h1_' + str(h1) + '_h2_' + str(h2) + '_epochs_' + str(
        num_epochs),
               processing_times[:, 0:slot])
    np.savetxt('load_poly_' + str(erasure_prob) + '_h1_' + str(h1) + '_h2_' + str(h2) + '_epochs_' + str(
        num_epochs) + '.txt',
               np.array(load_per_round))


def worker():
    # Receive straggling map
    straggling_map = np.empty([num_workers, approx_times])
    str_map_req = comm.Irecv(straggling_map, source=0, tag=1000)
    str_map_req.Wait()
    slot = 0
    while 1:
        # get status
        status = comm.recv(source=0, tag=0)
        if status == 'done':
            break
        # get size
        size = np.empty([2, 2], dtype=np.int)
        size_req = comm.Irecv(size, source=0, tag=0)
        size_req.Wait()
        # print(size)
        rec_A = np.empty(size[0, :], dtype=np.float64)
        rec_B = np.empty(size[1, :])

        req_rec_A = comm.Irecv(rec_A, source=0, tag=0)
        req_rec_B = comm.Irecv(rec_B, source=0, tag=1)
        req_rec_A.Wait()
        req_rec_B.Wait()
        init = time.time()
        result = rec_A.T @ rec_B
        if straggling_map[rank - 1, slot]:
            for _ in range(5-1):
                rec_A.T @ rec_B
        t = time.time() - init
        req_send_res = comm.Isend(result, dest=0, tag=10)
        req_send_res.Wait()
        comm.send(t, dest=0, tag=0)
        slot += 1




comm = MPI.COMM_WORLD
rank = comm.Get_rank()
erasure_prob = 0.1
num_epochs = 500
h1 = 3*1024
h2 = 3*1024
num_workers = 4
recov_threshold = 2
kA = first_divisor(recov_threshold)
kB = int(recov_threshold / first_divisor(recov_threshold))
approx_times = int(4 * 20 * 50 * 8 * 1.2)
channel_type = 'iid'
if rank == 0:
    master()
else:
    worker()
