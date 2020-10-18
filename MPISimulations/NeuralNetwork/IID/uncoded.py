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



def master():
    ### Initialization Phase
    # Creating Straggling map
    round_times = []
    processing_times = np.zeros([num_workers, num_epochs * 5 * 8 + 1000])
    slot = 0
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

    # Creating Models
    model1 = Model('MNIST', 1024, 0.1, h1, h2)
    model2 = Model('MNIST', 1024, 0.15, h1, h2)
    model3 = Model('MNIST', 1024, 0.2, h1, h2)
    model4 = Model('MNIST', 1024, 0.25, h1, h2)
    model5 = Model('MNIST', 1024, 0.3, h1, h2)
    models = [model1, model2, model3, model4, model5]
    model_under_operation = 0
    crt_job_per_model = np.zeros(len(models))
    straggling_map = np.zeros(num_workers)
    while 1:
        # determine whether there is anything to do
        crt_iteration = models[-1].iter_trained
        # print(crt_iteration)
        # print(model_under_operation)
        # print(crt_job_per_model[model_under_operation])
        if crt_iteration == num_epochs:
            status = 'done'
        else:
            status = 'work'
        for worker_idx in range(num_workers):
            comm.send(status, dest=worker_idx + 1, tag=0)
        if status == 'done':
            break
        if slot > 0:
            if channel_type == 'iid':
                straggling_map = np.random.binomial(1, erasure_prob, num_workers)
        for worker_idx in range(num_workers):
            comm.send(straggling_map[worker_idx], dest=worker_idx + 1, tag=0)
        # extract the job and divide
        if model_under_operation == len(models) - 1 and crt_job_per_model[model_under_operation] == 0:
            print(crt_iteration)
        A, B = models[model_under_operation].get_train_job(crt_job_per_model[model_under_operation])
        coded_A, coded_B = np.array_split(A, kA, axis=0), np.array_split(B, kB, axis=1)
        size_req = []
        for worker_idx in range(num_workers):
            size_req.append(
                comm.Isend(np.array([coded_A[worker_idx % kA].shape, coded_B[worker_idx // kA].shape], dtype=np.int)
                           , dest=worker_idx + 1, tag=0))
        MPI.Request.waitall(size_req)

        req_A = []
        req_B = []
        for worker_idx in range(num_workers):
            req_A.append(comm.Isend(np.ascontiguousarray(coded_A[worker_idx % kA]), dest=worker_idx + 1, tag=5))
            req_B.append(comm.Isend(np.ascontiguousarray(coded_B[worker_idx // kA]), dest=worker_idx + 1, tag=1))
        MPI.Request.waitall(req_A)
        MPI.Request.waitall(req_B)

        req_result = []
        result = [np.empty([coded_A[worker_idx % kA].shape[0], coded_B[worker_idx // kA].shape[1]]) for worker_idx in
                  range(num_workers)]
        for worker_idx in range(num_workers):
            req_result.append(comm.Irecv(result[worker_idx], source=worker_idx + 1, tag=10))
        MPI.Request.waitall(req_result)
        for worker_idx in range(num_workers):
            processing_times[worker_idx, slot] = comm.recv(source=worker_idx+1, tag=0)
        slot += 1
        output = np.concatenate([np.concatenate(result[kA * i:kA * i + kA], axis=0) for i in range(kB)], axis=1)
        models[model_under_operation].push_train_job(crt_job_per_model[model_under_operation], output)
        crt_job_per_model[model_under_operation] += 1
        crt_job_per_model[model_under_operation] = crt_job_per_model[model_under_operation] % 8
        model_under_operation += 1
        model_under_operation = model_under_operation % len(models)
    train_acc = []
    test_acc = []
    result = []  # train_acc, train_loss, test_acc, test_loss
    for mdl in models:
        result.append(mdl.performance())
    for i in range(5):
        train_acc.append(models[i].train_acc)
        test_acc.append(models[i].test_acc)
    np.savetxt(
        'train_acc_per_iteration_uncoded_p_' + str(erasure_prob) + '_h1_' + str(h1) + '_h2_' + str(
            h2) + '_epochs_' + str(
            num_epochs) + '.txt',
        np.array(train_acc))
    np.savetxt(
        'test_acc_per_iteration_uncoded_p_' + str(erasure_prob) + '_h1_' + str(h1) + '_h2_' + str(h2) + '_epochs_' + str(
            num_epochs) + '.txt',
        np.array(test_acc))
    np.savetxt(
        'result_uncoded_' + str(erasure_prob) + '_h1_' + str(h1) + '_h2_' + str(h2) + '_epochs_' + str(num_epochs),
        np.array(result))
    np.savetxt(
        'processing_time_uncoded_p_' + str(erasure_prob) + '_h1_' + str(h1) + '_h2_' + str(h2) + '_epochs_' + str(
            num_epochs),
        processing_times[:, 0:slot])


def worker():
    # Receive straggling map
    slot = 0
    while 1:
        # get status
        status = comm.recv(source=0, tag=0)
        if status == 'done':
            break
        # get size
        straggling_status = comm.recv(source=0, tag=0)
        size = np.empty([2, 2], dtype=np.int)
        size_req = comm.Irecv(size, source=0, tag=0)
        size_req.Wait()
        # print(size)
        rec_A = np.empty(size[0, :], dtype=np.float64)
        rec_B = np.empty(size[1, :])

        req_rec_A = comm.Irecv(rec_A, source=0, tag=5)
        req_rec_B = comm.Irecv(rec_B, source=0, tag=1)
        req_rec_A.Wait()
        req_rec_B.Wait()
        init = time.time()
        result = rec_A @ rec_B
        if straggling_status == 1:
            for _ in range(5-1):
                rec_A @ rec_B
        t = time.time() - init
        req_send_res = comm.Isend(result, dest=0, tag=10)
        req_send_res.Wait()
        comm.send(t, dest=0, tag=0)
        slot += 1


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
erasure_prob = 0
num_epochs = 500
h1 = 32
h2 = 32
num_workers = 2
kA = first_divisor(num_workers)
kB = int(num_workers / first_divisor(num_workers))
approx_times = int(4 * 20 * 50 * 8 * 1.2)
channel_type = 'iid'
if rank == 0:
    master()
else:
    worker()
