import time

import numpy as np
import tensorflow as tf
from mpi4py import MPI
from scipy.special import logsumexp


def first_divisor(n):
    for i in range(2, n + 1):
        if n % i == 0:
            return i


def window_check(window):
    if np.sum(window) > N * epsilon:
        return 0
    return 1


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
        self.train_loss = []
        self.test_acc = []
        self.test_loss = []
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
            train_acc, train_loss, test_acc, test_loss = self.performance()
            self.train_acc.append(train_acc)
            self.train_loss.append(train_loss)
            self.test_acc.append(test_acc)
            self.test_loss.append(test_loss)
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
    node_points = np.random.rand(n).astype(np.float64)*2-1
    Coding_matrix1 = np.zeros((n, k))
    for j in range(0, n):
        Coding_matrix1[j, :] = (node_points[j]) ** np.arange(k)
    Coding_A = Coding_matrix1[:, ::kB]
    W1a = np.hsplit(A, kA)
    W2a = []
    (uu, vv) = np.shape(W1a[0])
    for i in range(0, n):
        W2a.append(np.zeros((uu, vv)))
        for j in range(0, kA):
            W2a[i] = W2a[i] + Coding_A[i, j] * W1a[j]
    Coding_B = Coding_matrix1[:, 0:kB]
    W1b = np.hsplit(B, kB)
    W2b = []
    (uu, vv) = np.shape(W1b[0])
    for i in range(0, n):
        W2b.append(np.zeros((uu, vv)))
        for j in range(0, kB):
            W2b[i] = W2b[i] + Coding_B[i, j] * W1b[j]
    return [W2a, W2b], node_points


def poly_decode(A_size, B_size, kA, kB, worker_product, uncoded_pieces, nodes):
    ## uncoded pieces are passed in order of A0B0, A0B1, A0B2, ...
    [t, r] = A_size
    [t, w] = B_size
    r_p = r // kA
    c_p = w // kB
    reduced_products = worker_product
    if len(uncoded_pieces) > 0:
        reduced_products = []
        for prod_idx in range(len(worker_product)):
            diff = np.sum([uncoded_pieces[i] * (nodes[prod_idx] ** i) for i in range(len(uncoded_pieces))],
                          axis=0)
            reduced_products.append((worker_product[prod_idx] - diff) / (nodes[prod_idx] ** len(uncoded_pieces)))
    if len(worker_product) > 0:
        c = r_p
        d = c_p
        k = kA * kB - len(uncoded_pieces)
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
        if idx < len(uncoded_pieces):
            to_go = uncoded_pieces[idx]
        else:
            to_go = np.reshape(decoded_blocks[:, idx - len(uncoded_pieces)], (g, h))
        final_res[row * r_p:row * r_p + r_p, col * c_p:col * c_p + c_p] = to_go

    return final_res


def sorted_unnest(arr):
    # returns the longest consecutive inner array
    if len(arr) == 0:
        return []
    # print(arr)
    for i in range(max(arr)):
        if i in arr:
            continue
        else:
            return list(range(i))
    return list(range(len(arr)))


def master():
    ### Initialization Phase
    # Creating Straggling map
    # print('Master Awake')
    # Creating Models
    # print('Creating Models')

    model1 = Model('MNIST', 1024, 0.1, h1, h2)
    model2 = Model('MNIST', 1024, 0.15, h1, h2)
    model3 = Model('MNIST', 1024, 0.2, h1, h2)
    model4 = Model('MNIST', 1024, 0.25, h1, h2)
    model5 = Model('MNIST', 1024, 0.3, h1, h2)
    models = [model1, model2, model3, model4, model5]
    model_under_operation = 0
    crt_job_per_model = np.zeros(len(models))
    queue = []
    straggling_history = np.zeros([num_workers, approx_times])
    processing_times = np.zeros([num_workers, approx_times])
    slot = 0
    encoding_times = []
    decoding_times = []
    load_per_round = []
    # print('Starting Training')
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
        decode_enable = False

        if slot > 0:
            if len(queue[0]['coded_collected_pieces']) >= S:
                to_decode = queue.pop(0)
                decode_enable = True
            if channel_type == 'iid':
                straggling_map = np.random.binomial(1, erasure_prob, num_workers)
        # extract the job and divide
        A, B = models[model_under_operation].get_train_job(crt_job_per_model[model_under_operation])
        # pad zeros to fix any size problem
        A_shape_orig, B_shape_orig = np.shape(A), np.shape(B)
        new_A, new_B = np.copy(A), np.copy(B)
        if A_shape_orig[0] % kA != 0:
            new_A = np.zeros([A_shape_orig[0] + kA - (A_shape_orig[0] % kA), A_shape_orig[1]])
            new_A[0:A_shape_orig[0], :] = A
        if B_shape_orig[1] % kB != 0:
            new_B = np.zeros([B_shape_orig[0], B_shape_orig[1] + kB - (B_shape_orig[1] % kB)])
            new_B[:, 0:B_shape_orig[1]] = B
        job = {'A_pieces': np.hsplit(new_A.T, kA), 'B_pieces': np.hsplit(new_B, kB),
               'time_in_queue': 0, 'coded_collected_pieces': [], 'collected_eval_points': [],
               'A_size': np.shape(new_A.T), 'B_size': np.shape(new_B),
               'res_shape': [A.shape[0], B.shape[1]],
               'model': model_under_operation,
               'piece_idx': crt_job_per_model[model_under_operation],
               'A': new_A.T, 'B': new_B}
        queue.append(job)
        # Send out jobs
        # tell workers what to expect
        # print('Sending out Numbers')
        crt_place_map = [{} for _ in
                         range(num_workers)]
        buffer_numbers = []
        for worker_idx in range(num_workers):
            numbers = {}
            for job_idx in range(len(queue)):
                job = queue[job_idx]
                if len(job['coded_collected_pieces']) >= S:
                    numbers[job_idx] = 0
                    crt_place_map[worker_idx][job_idx] = None
                    continue
                if job['time_in_queue'] == T:
                    remaining_pieces = S - len(job['coded_collected_pieces'])
                    numbers[job_idx] = int(np.ceil(remaining_pieces / lam))
                    crt_place_map[worker_idx][job_idx] = [int(np.ceil(remaining_pieces / lam)), 'c']
                if job['time_in_queue'] < T:
                    crt_place_map[worker_idx][job_idx] = [1, 'c']
                    numbers[job_idx] = 1

            numbers['total'] = len(queue)
            comm.send(numbers, dest=worker_idx + 1, tag=0)
            buffer_numbers.append(numbers)
        # print('buffer_numbers', buffer_numbers)
        # tell workers the shape of pieces
        # print('Send out shapes')
        shape_send = []
        buffer_shape = []
        for worker_idx in range(num_workers):
            shapes = []
            shapes_dict = {}
            for job_idx in range(len(queue)):
                job = queue[job_idx]
                A_piece_shape = job['A_pieces'][0].shape
                B_pieces_shape = job['B_pieces'][0].shape
                if crt_place_map[worker_idx][job_idx] == None:
                    continue
                if isinstance(crt_place_map[worker_idx][job_idx], list):
                    number = crt_place_map[worker_idx][job_idx][0]
                    shapes_dict[job_idx] = np.array([[A_piece_shape[0], A_piece_shape[1] * number],
                                                     [B_pieces_shape[0], B_pieces_shape[1] * number]], dtype=int)
                    shapes.append(np.array([[A_piece_shape[0], A_piece_shape[1] * number],
                                            [B_pieces_shape[0], B_pieces_shape[1] * number]], dtype=int))
                    continue
            buffer_shape.append(shapes_dict)
            shape_send.append(comm.Isend(np.hstack(shapes), dest=worker_idx + 1, tag=0))
        MPI.Request.waitall(shape_send)
        # Send out actual pieces
        # print('send out pieces')
        send_req = []
        eval_points = [{} for _ in range(num_workers)]
        enc_time = 0

        for worker_idx in range(num_workers):
            load = 0
            for job_idx in range(len(queue)):
                job = queue[job_idx]
                if crt_place_map[worker_idx][job_idx] == None:
                    continue
                if isinstance(crt_place_map[worker_idx][job_idx], list):
                    init_enc = time.time()
                    [coded_A, coded_B], eval_points_worker = poly_encode(job['A'], job['B'], kA, kB,
                                                                         crt_place_map[worker_idx][job_idx][0])
                    shapeA = np.shape(coded_A[0])
                    shapeB = np.shape(coded_B[0])
                    load += shapeA[0]*shapeA[1]*shapeB[1]*len(coded_A)
                    enc_time += time.time() - init_enc
                    eval_points[worker_idx][job_idx] = eval_points_worker
                    if crt_place_map[worker_idx][job_idx][0] > 1:
                        coded_A = np.concatenate(coded_A, axis=1)
                        coded_B = np.concatenate(coded_B, axis=1)
                    else:
                        coded_A = coded_A[0]
                        coded_B = coded_B[0]
                    send_req.append(comm.Isend(np.ascontiguousarray(coded_A), dest=worker_idx + 1, tag=job_idx))
                    send_req.append(comm.Isend(np.ascontiguousarray(coded_B), dest=worker_idx + 1, tag=job_idx))
        load_per_round.append(load)
        encoding_times.append(enc_time)
        MPI.Request.waitall(send_req)
        # Setup receive buffers and wait instructions
        # Do intermediate Processings while waiting TODO
        if decode_enable:
            init_dec = time.time()
            recon = poly_decode(to_decode['A_size'], to_decode['B_size'],
                                kA, kB,
                                to_decode['coded_collected_pieces'][0:S], [],
                                to_decode['collected_eval_points'][0:S])
            recon = recon[0:to_decode['res_shape'][0], 0:to_decode['res_shape'][1]]

            models[to_decode['model']].push_train_job(to_decode['piece_idx'], recon)
            crt_job_per_model[to_decode['model']] += 1
            crt_job_per_model[to_decode['model']] = crt_job_per_model[to_decode['model']] % 8
            decoding_times.append(time.time() - init_dec)
        rec_buffer = []
        rec_request = []
        rec_request_map = []
        for worker_idx in range(num_workers):
            numbers = buffer_numbers[worker_idx]
            shapes = buffer_shape[worker_idx]
            result_buffer = {}
            for job_idx in range(len(queue)):
                if numbers[job_idx] == 0:
                    continue
                crt_shape = shapes[job_idx]
                result_buffer[job_idx] = np.empty([crt_shape[0, 1] // numbers[job_idx], crt_shape[1, 1]])
            rec_buffer.append(result_buffer)
            for job_idx in range(len(queue)):
                if numbers[job_idx] > 0:
                    rec_request.append(comm.Irecv(rec_buffer[worker_idx][job_idx], source=worker_idx + 1, tag=job_idx))
                    rec_request_map.append(worker_idx)
        MPI.Request.waitall(rec_request)

        # record receiving times
        for worker_idx in range(num_workers):
            processing_times[worker_idx, slot] = comm.recv(source=worker_idx + 1, tag=0)
        # print('Jobs Rec')
        # determine straggling status of workers
        times_sorted_idx = np.argsort(processing_times[:, slot])
        best_time = processing_times[times_sorted_idx[0], slot]
        for worker_idx in times_sorted_idx:
            if processing_times[worker_idx, slot] > (1 + tol) * best_time:
                straggling_history[worker_idx, slot] = 1
        # print(straggling_history[:, 0:slot + 1])
        for idx in range(len(times_sorted_idx)):
            worker_idx = times_sorted_idx[idx]
            result = rec_buffer[worker_idx]
            numbers = buffer_numbers[worker_idx]
            if straggling_history[worker_idx, slot] == 0:
                for job_idx in range(len(queue)):
                    if numbers[job_idx] == 1:
                        if isinstance(crt_place_map[worker_idx][job_idx], list):
                            queue[job_idx]['coded_collected_pieces'].append(result[job_idx])
                            queue[job_idx]['collected_eval_points'].append(eval_points[worker_idx][job_idx][0])
                    if numbers[job_idx] > 1:
                        list_ = np.hsplit(result[job_idx], numbers[job_idx])
                        for idx in range(len(list_)):
                            queue[job_idx]['coded_collected_pieces'].append(list_[idx])
                            queue[job_idx]['collected_eval_points'].append(eval_points[worker_idx][job_idx][idx])
            if straggling_history[worker_idx, slot] == 1:
                important_job = queue[0]
                if important_job['time_in_queue'] < T:
                    break
                if len(
                        important_job['coded_collected_pieces']) >= S:
                    break
                else:
                    for job_idx in range(len(queue)):
                        if numbers[job_idx] == 1:
                            if isinstance(crt_place_map[worker_idx][job_idx], list):
                                queue[job_idx]['coded_collected_pieces'].append(result[job_idx])
                                queue[job_idx]['collected_eval_points'].append(eval_points[worker_idx][job_idx][0])
                        if numbers[job_idx] > 1:
                            list_ = np.hsplit(result[job_idx], numbers[job_idx])
                            for idx in range(len(list_)):
                                queue[job_idx]['coded_collected_pieces'].append(list_[idx])
                                queue[job_idx]['collected_eval_points'].append(eval_points[worker_idx][job_idx][idx])
                    straggling_history[worker_idx, slot] = 0
                    straggling_history[times_sorted_idx[idx:], slot] = 0
        slot += 1
        for job_idx in range(len(queue)):
            queue[job_idx]['time_in_queue'] += 1
        model_under_operation += 1
        model_under_operation = model_under_operation % len(models)
        total = 0
        for job_idx in range(len(queue)):
            total += buffer_numbers[0][job_idx]
        # print(crt_place_map)
        # print(prev_place_map)
        # print(straggling_history[:, 0:slot])
    result = []  # train_acc, train_loss, test_acc, test_loss
    for mdl in models:
        result.append(mdl.performance())
    print(encoding_times[0:10])
    print(load_per_round[0:10])
    print(decoding_times[0:10])
    print(processing_times[0:10])
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    for i in range(5):
        train_acc.append(models[i].train_acc)
        train_loss.append(models[i].train_loss)
        test_acc.append(models[i].test_acc)
        test_loss.append(models[i].test_loss)
    np.savetxt(
        'load_per_round_MI_' + 'fritch' + '_h1_' + str(h1) + '_h2_' + str(h2) + '_epochs_' + str(
            num_epochs) + '.txt',
        np.array(load_per_round))


    np.savetxt(
        'train_acc_per_iteration_MI_' + 'fritch' + '_h1_' + str(h1) + '_h2_' + str(h2) + '_epochs_' + str(
            num_epochs) + '.txt',
        np.array(train_acc))
    np.savetxt(
        'train_loss_per_iteration_MI_' + 'fritch' + '_h1_' + str(h1) + '_h2_' + str(h2) + '_epochs_' + str(
            num_epochs) + '.txt',
        np.array(train_loss))
    np.savetxt(
        'test_acc_per_iteration_MI_' + 'fritch' + '_h1_' + str(h1) + '_h2_' + str(h2) + '_epochs_' + str(
            num_epochs) + '.txt',
        np.array(test_acc))
    np.savetxt(
        'test_loss_per_iteration_MI_' + 'fritch' + '_h1_' + str(h1) + '_h2_' + str(h2) + '_epochs_' + str(
            num_epochs) + '.txt',
        np.array(test_loss))
    np.savetxt(
        'result_MI_' + 'fritch' + '_h1_' + str(h1) + '_h2_' + str(h2) + '_epochs_' + str(num_epochs) + '.txt',
        np.array(result))
    np.savetxt('straggling_history_MI_' + 'fritch' + '_h1_' + str(h1) + '_h2_' + str(h2) + '_epochs_' + str(
        num_epochs) + '.txt',
               straggling_history[:, 0:slot])
    np.savetxt('encoding_time_MI_' + 'fritch' + '_h1_' + str(h1) + '_h2_' + str(h2) + '_epochs_' + str(
        num_epochs) + '.txt',
               np.array(encoding_times))
    np.savetxt('decoding_time_MI_' + 'fritch' + '_h1_' + str(h1) + '_h2_' + str(h2) + '_epochs_' + str(
        num_epochs) + '.txt',
               np.array(decoding_times))
    np.savetxt('processing_time_MI_' + 'fritch' + '_h1_' + str(h1) + '_h2_' + str(h2) + '_epochs_' + str(
        num_epochs) + '.txt',
               processing_times[:, 0:slot])


def worker():
    slot = 0
    state = 0
    while 1:
        if channel_type == 'iid':
            straggling_status = np.random.binomial(1, erasure_prob)
        if channel_type == 'fritch':
            # determine straggling status
            if state == 0:
                straggling_status = 0
            else:
                straggling_status = 1
            # change state
            if state == 0:
                if np.random.binomial(1, a):
                    state = 1
            elif state == 1:
                if np.random.binomial(1, b):
                    state = 2
            elif state == 2:
                if np.random.binomial(1, b):
                    state = 0
        # get status
        status = comm.recv(source=0, tag=0)
        if status == 'done':
            break
        # get length
        numbers = comm.recv(source=0, tag=0)
        # print(numbers)
        pieces_to_expect = 0
        for job_idx in range(numbers['total']):
            if numbers[job_idx] > 0:
                pieces_to_expect += 1
        # Create Buffer to get shapes
        shape_buff = np.empty([2, 2 * pieces_to_expect], dtype=int)
        buff_recv_req = comm.Irecv(shape_buff, source=0, tag=0)
        buff_recv_req.Wait()
        shapes = np.hsplit(shape_buff, pieces_to_expect)
        # print(shapes)
        rec_buffer_A = [np.empty(shape[0, :]) for shape in shapes]
        rec_buffer_B = [np.empty(shape[1, :]) for shape in shapes]
        rec_req = []
        cntr = 0
        for job_idx in range(numbers['total']):
            if numbers[job_idx] != 0:
                rec_req.append(comm.Irecv(rec_buffer_A[cntr], source=0, tag=job_idx))
                rec_req.append(comm.Irecv(rec_buffer_B[cntr], source=0, tag=job_idx))
                cntr += 1
        # print(len(rec_req))
        MPI.Request.waitall(rec_req)
        init = time.time()
        to_do_A = []
        to_do_B = []
        cntr = 0
        for job_idx in range(numbers['total']):
            if numbers[job_idx] == 0:
                continue
            if numbers[job_idx] == 1:
                to_do_A.append(rec_buffer_A[cntr])
                to_do_B.append(rec_buffer_B[cntr])
                cntr += 1
            if numbers[job_idx] > 1:
                pieces_A = np.hsplit(rec_buffer_A[cntr], numbers[job_idx])
                pieces_B = np.hsplit(rec_buffer_B[cntr], numbers[job_idx])
                for idx in range(len(pieces_A)):
                    to_do_A.append(pieces_A[idx])
                    to_do_B.append(pieces_B[idx])
                cntr += 1
        res = [to_do_A[i].T @ to_do_B[i] for i in range(len(to_do_A))]
        if straggling_status == 1:
            for _ in range(5 - 1):
                res = [to_do_A[i].T @ to_do_B[i] for i in range(len(to_do_A))]
        proc_time = time.time() - init
        # return results
        cntr = 0
        send_req = []
        for job_idx in range(numbers['total']):
            if numbers[job_idx] == 0:
                continue
            if numbers[job_idx] == 1:
                send_req.append(comm.Isend(np.ascontiguousarray(res[cntr]), dest=0, tag=job_idx))
                cntr += 1
            if numbers[job_idx] > 1:
                send_req.append(
                    comm.Isend(np.ascontiguousarray(np.hstack(res[cntr:cntr + numbers[job_idx]])), dest=0, tag=job_idx))
                cntr += numbers[job_idx]
        MPI.Request.waitall(send_req)
        # return processing time
        comm.send(proc_time, dest=0, tag=0)
        slot += 1


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
h1 = 32
h2 = 32
num_epochs = 100
erasure_prob = 0.1
num_workers = 2
tol = 0.9
T = 3
S = 4
kA = first_divisor(S)
kB = int(S / kA)
approx_times = int(num_epochs*8*(T+2)*1.5)
channel_type = 'fritch'
a = 0.8
b = 0.5
lam = 2
if rank == 0:
    master()
else:
    worker()
