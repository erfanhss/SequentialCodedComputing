import numpy as np

def generate_map(P, length, alpha, beta, N=1):
  res = np.zeros([P, length])
  for worker_idx in range(P):
    crt_state = 0
    for slot in range(length):
      if slot == 0:
        crt_state = 0
        continue
      if crt_state == 0:
        if np.random.binomial(1, alpha):
          crt_state =1
          res[worker_idx, slot] = 1
        else:
          crt_state=0
          res[worker_idx, slot] = 0
      else:
        if np.random.binomial(1, beta):
          crt_state = (crt_state + 1) % (N+1)
          res[worker_idx, slot] = 1
          if crt_state == 0:
            res[worker_idx, slot] = 0
        else:
          crt_state = crt_state
          res[worker_idx, slot] = 1
  return res

def check_window_bursty(window, B, epsilon):
  num = 0
  for i in range(np.shape(window)[0]):
    row = window[i, :]
    idx = np.where(row==1)[0]
    # print(idx)
    if len(idx) == 0:
      continue
    num += 1
    burst_len = idx[-1] - idx[0] + 1
    if burst_len > B:
      return 0
  # print('num', num)
  if num > epsilon:
    return 0
  return 1

def check_window_arbitrary(window, N):
  if np.sum(window) > N:
    return 0
  return 1

def DIP_fritchman(J, P, T, N, a, b, states=1, alpha=5):
  S = (T+1)*P-N
  queue = []
  slot = 0
  delivered = 0
  straggling_map = generate_map(P, J*2, a, b, N=states)
  straggling_history = np.zeros_like(straggling_map)
  runtime = []
  while 1:
    if delivered == J:
      break
    if slot > 0:
      if queue[0]['pieces_rec'] >= S:
        delivered += 1
        queue.pop(0)
    job = {'pieces_rec': 0, 'time_in_queue':0}
    queue.append(job)
    numbers = {}
    crt_load = 0
    for job_idx in range(len(queue)):
      job = queue[job_idx]
      if job['pieces_rec'] >= S:
        numbers[job_idx] = 0
      else: 
        numbers[job_idx] = 1
        crt_load += 1
    wait = 0
    straggling_history[:, slot] = straggling_map[:, slot]
    if np.sum(straggling_history[:, slot]) == P:
      straggling_history[:, slot] = 0
      wait = 1
    list_crt = np.argsort(straggling_history[:, slot])
    
    for idx in list_crt:
      worker_idx = list_crt[idx]
      if straggling_history[worker_idx, slot] == 0:
        for job_idx in range(len(queue)):
          queue[job_idx]['pieces_rec'] += numbers[job_idx]
      else:
        window = straggling_history[:, max(0, slot-T):slot+1]
        if check_window_arbitrary(window, N) == 1:
          break
        else:
          straggling_history[worker_idx, slot] = 0
          wait = 1
          for job_idx in range(len(queue)):
            queue[job_idx]['pieces_rec'] += numbers[job_idx]
    crt_runtime = crt_load * (1+(alpha-1)*wait)/S
    slot += 1
    for job_idx in range(len(queue)):
      queue[job_idx]['time_in_queue'] += 1
    runtime.append(crt_runtime)
  return np.sum(runtime)/J


def DIP_IID(J, P, T, N, p, alpha=5):
  S = (T+1)*P-N
  queue = []
  slot = 0
  delivered = 0
  straggling_map = np.random.binomial(1, p, [P, J*2])
  straggling_history = np.zeros_like(straggling_map)
  runtime = []
  while 1:
    if delivered == J:
      break
    if slot > 0:
      if queue[0]['pieces_rec'] >= S:
        delivered += 1
        queue.pop(0)
    job = {'pieces_rec': 0, 'time_in_queue':0}
    queue.append(job)
    numbers = {}
    crt_load = 0
    for job_idx in range(len(queue)):
      job = queue[job_idx]
      if job['pieces_rec'] >= S:
        numbers[job_idx] = 0
      else: 
        numbers[job_idx] = 1
        crt_load += 1
    wait = 0
    straggling_history[:, slot] = straggling_map[:, slot]
    if np.sum(straggling_history[:, slot]) == P:
      straggling_history[:, slot] = 0
      wait = 1
    list_crt = np.argsort(straggling_history[:, slot])
    
    for idx in list_crt:
      worker_idx = list_crt[idx]
      if straggling_history[worker_idx, slot] == 0:
        for job_idx in range(len(queue)):
          queue[job_idx]['pieces_rec'] += numbers[job_idx]
      else:
        window = straggling_history[:, max(0, slot-T):slot+1]
        if check_window_arbitrary(window, N) == 1:
          break
        else:
          straggling_history[worker_idx, slot] = 0
          wait = 1
          for job_idx in range(len(queue)):
            queue[job_idx]['pieces_rec'] += numbers[job_idx]
    crt_runtime = crt_load * (1+(alpha-1)*wait)/S
    slot += 1
    for job_idx in range(len(queue)):
      queue[job_idx]['time_in_queue'] += 1
    runtime.append(crt_runtime)
  return np.sum(runtime)/J

def poly_fritch(J, S, P, alpha, states, a, b):
  straggling_map = generate_map(P, J, a, b, N=states)
  round_times = []
  for j in range(J):
    if np.sum(straggling_map[:, j]) <= S:
      round_times.append(1/(P-S))
    else:
      round_times.append(alpha/(P-S))
  return np.sum(round_times)/ J


def IDIP_iid(J, T, P, S, lam, p, alpha=5):
  queue = []
  slot = 0
  delivered = 0
  straggling_map = np.random.binomial(1, p, [P, J*2])
  straggling_history = np.zeros_like(straggling_map)
  runtime = []
  while 1:
    if delivered == J:
      break
    if slot > 0:
      if queue[0]['pieces_rec'] >= S:
        delivered += 1
        queue.pop(0)
    job = {'pieces_rec': 0, 'time_in_queue':0}
    queue.append(job)
    numbers = {}
    crt_load = 0
    for job_idx in range(len(queue)):
      job = queue[job_idx]
      if job['pieces_rec'] >= S:
        numbers[job_idx] = 0
        continue
      if job['time_in_queue'] < T:
        numbers[job_idx] = 1
        crt_load += 1
        continue
      if job['time_in_queue'] == T:
        rem = np.ceil((S - job['pieces_rec'])/lam)
        numbers[job_idx] = rem
        crt_load += rem
    wait = 0
    straggling_history[:, slot] = straggling_map[:, slot]
    if np.sum(straggling_history[:, slot]) == P:
      straggling_history[:, slot] = 0
      wait = 1
    list_crt = np.argsort(straggling_history[:, slot])
    for idx in list_crt:
      worker_idx = list_crt[idx]
      if straggling_history[worker_idx, slot] == 0:
        for job_idx in range(len(queue)):
          queue[job_idx]['pieces_rec'] += numbers[job_idx]
      else:
        if queue[0]['pieces_rec'] < S and queue[0]['time_in_queue'] == T:
          straggling_history[list_crt[idx:], slot] = 0
          wait = 1
          for job_idx in range(len(queue)):
            queue[job_idx]['pieces_rec'] += numbers[job_idx]
    crt_runtime = crt_load * (1+(alpha-1)*wait)/S
    slot += 1
    for job_idx in range(len(queue)):
      queue[job_idx]['time_in_queue'] += 1
    runtime.append(crt_runtime)
  return np.sum(runtime)/J



def DIP_fritchman(J, T, P, S, lam, a, b, states=1, alpha=5):
  queue = []
  slot = 0
  delivered = 0
  straggling_map = generate_map(P, J*2, a, b, states)
  straggling_history = np.zeros_like(straggling_map)
  runtime = []
  while 1:
    if delivered == J:
      break
    if slot > 0:
      if queue[0]['pieces_rec'] >= S:
        delivered += 1
        queue.pop(0)
    job = {'pieces_rec': 0, 'time_in_queue':0}
    queue.append(job)
    numbers = {}
    crt_load = 0
    for job_idx in range(len(queue)):
      job = queue[job_idx]
      if job['pieces_rec'] >= S:
        numbers[job_idx] = 0
        continue
      if job['time_in_queue'] < T:
        numbers[job_idx] = 1
        crt_load += 1
        continue
      if job['time_in_queue'] == T:
        rem = np.ceil((S - job['pieces_rec'])/lam)
        numbers[job_idx] = rem
        crt_load += rem
    
    wait = 0
    straggling_history[:, slot] = straggling_map[:, slot]
    if np.sum(straggling_history[:, slot]) == P:
      straggling_history[:, slot] = 0
      wait = 1
    list_crt = np.argsort(straggling_history[:, slot])
    for idx in list_crt:
      worker_idx = list_crt[idx]
      if straggling_history[worker_idx, slot] == 0:
        for job_idx in range(len(queue)):
          queue[job_idx]['pieces_rec'] += numbers[job_idx]
      else:
        if queue[0]['pieces_rec'] < S and queue[0]['time_in_queue'] == T:
          straggling_history[list_crt[idx:], slot] = 0
          wait = 1
          for job_idx in range(len(queue)):
            queue[job_idx]['pieces_rec'] += numbers[job_idx]
    crt_runtime = crt_load * (1+(alpha-1)*wait)/S
    slot += 1
    for job_idx in range(len(queue)):
      queue[job_idx]['time_in_queue'] += 1
    runtime.append(crt_runtime)
  return np.sum(runtime)/J

  
def IDIP_fritchman(J, P, W, B, epsilon, a, b, states=1, alpha=5):
  S = (W-1+B)*P - B*epsilon
  queue = []
  slot = 0
  delivered = 0
  straggling_map = generate_map(P, J*2, a, b, states)
  straggling_history = np.zeros_like(straggling_map)
  runtime = []
  while 1:
    if delivered == J:
      break
    if slot > 0:
      if queue[0]['pieces_rec'] >= S:
        delivered += 1
        queue.pop(0)
    job = {'pieces_rec': 0, 'time_in_queue':0}
    queue.append(job)
    numbers = {}
    crt_load = 0
    for job_idx in range(len(queue)):
      job = queue[job_idx]
      if job['pieces_rec'] >= S:
        numbers[job_idx] = 0
      else: 
        numbers[job_idx] = 1
        crt_load += 1
    wait = 0
    straggling_history[:, slot] = straggling_map[:, slot]
    if np.sum(straggling_history[:, slot]) == P:
      straggling_history[:, slot] = 0
      wait = 1
    list_crt = np.argsort(straggling_history[:, slot])
    for idx in list_crt:
      worker_idx = list_crt[idx]
      if straggling_history[worker_idx, slot] == 0:
        for job_idx in range(len(queue)):
          queue[job_idx]['pieces_rec'] += numbers[job_idx]
      else:
        window = straggling_history[:, max(0, slot-W+1):slot+1]
        if check_window_bursty(window, B, epsilon) == 1:
          break
        else:
          straggling_history[worker_idx, slot] = 0
          wait = 1
          for job_idx in range(len(queue)):
            queue[job_idx]['pieces_rec'] += numbers[job_idx]
    crt_runtime = crt_load * (1+(alpha-1)*wait)/S
    slot += 1
    for job_idx in range(len(queue)):
      queue[job_idx]['time_in_queue'] += 1
    runtime.append(crt_runtime)
  return np.sum(runtime)/J
