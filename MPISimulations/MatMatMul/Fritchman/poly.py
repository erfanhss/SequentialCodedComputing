import time
import numpy as np
from mpi4py import MPI


def first_divisor(n):
    for i in range(2, n + 1):
        if n % i == 0:
            return i


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
    processing_times = np.zeros([num_workers, num_jobs+1000])
    slot = 0
    encoding_times = []
    decoding_times = []
    # Creating Models
    # Creating Models
    crt_job = 0
    decode = 0
    load_per_round = []
    while 1:
        # determine whether there is anything to do
        print(crt_job)
        if crt_job == num_jobs:
            status = 'done'
        else:
            status = 'work'
        for worker_idx in range(num_workers):
            comm.send(status, dest=worker_idx + 1, tag=0)
        if status == 'done':
            break
        # extract the job and divide
        init_enc = time.time()
        A, B = np.random.random(A_dim), np.random.random(B_dim)

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
            load = shapeA[0] * shapeA[1] * shapeB[1]
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
        crt_job += 1
    print(encoding_times[0:10])
    print(decoding_times[0:10])
    print(processing_times[0:10])
    np.savetxt('encoding_time_poly_a_' + str(a) + '_b_' + str(b),
               np.array(encoding_times))
    np.savetxt('decoding_time_poly_a_' + str(a) + '_b_' + str(b),
               np.array(decoding_times))
    np.savetxt('processing_time_poly_a_' + str(a) + '_b_' + str(b),
               processing_times[:, 0:slot])
    np.savetxt('load_poly_a_' + str(a) + '_b_' + str(b),
               np.array(load_per_round))

def worker():
    # Receive straggling map
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
        if straggling_status:
            for _ in range(5-1):
                rec_A.T @ rec_B
        t = time.time() - init
        req_send_res = comm.Isend(result, dest=0, tag=10)
        req_send_res.Wait()
        comm.send(t, dest=0, tag=0)
        slot += 1




comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_jobs = 300
A_dim, B_dim = [4000, 4000], [4000, 4000]
num_workers = 4
recov_threshold = 2
kA = first_divisor(recov_threshold)
kB = int(recov_threshold / first_divisor(recov_threshold))
channel_type = 'fritch'
a = 0.2
b = 0.2
if rank == 0:
    master()
else:
    worker()
