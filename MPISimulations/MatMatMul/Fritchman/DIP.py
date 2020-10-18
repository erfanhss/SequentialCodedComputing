import time

import numpy as np
from mpi4py import MPI


def first_divisor(n):
    for i in range(2, n + 1):
        if n % i == 0:
            return i


def window_check(window):
    if np.sum(window) > N * epsilon:
        return 0
    return 1


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
    crt_job = 0
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
        print(crt_job)
        if crt_job == num_jobs:
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
        A, B = np.random.random(A_dim), np.random.random(B_dim)
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

        total = 0
        for job_idx in range(len(queue)):
            total += buffer_numbers[0][job_idx]
        # print(crt_place_map)
        # print(prev_place_map)
        # print(straggling_history[:, 0:slot])
        crt_job += 1
    np.savetxt(
        'load_per_round_DIP_' + 'fritch_a_' + str(a) + '_b_' + str(b),
        np.array(load_per_round))
    np.savetxt('straggling_history_DIP_' + 'fritch_a_' + str(a) + '_b_' + str(b),
               straggling_history[:, 0:slot])
    np.savetxt('encoding_time_DIP_' + 'fritch_a_' + str(a) + '_b_' + str(b),
               np.array(encoding_times))
    np.savetxt('decoding_time_DIP_' + 'fritch_a_' + str(a) + '_b_' + str(b),
               np.array(decoding_times))
    np.savetxt('processing_time_DIP_' + 'fritch_a_' + str(a) + '_b_' + str(b),
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
num_jobs = 300
A_dim, B_dim = [4000, 4000], [4000, 4000]
num_workers = 4
tol = 0.9
T = 4
S = 4
kA = first_divisor(S)
kB = int(S / kA)
approx_times = int(num_jobs*1.5)
channel_type = 'fritch'
a = 0.2
b = 0.2
lam = 3
if rank == 0:
    master()
else:
    worker()
