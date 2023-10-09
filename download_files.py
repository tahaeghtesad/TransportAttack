import os
from datetime import datetime

import numpy as np
import paramiko
from tqdm import tqdm


if __name__ == '__main__':
    hostkeys = paramiko.hostkeys.HostKeys(filename='/Users/txe5135/.ssh/known_hosts')
    fingerprint = hostkeys.lookup('ssh.ist.psu.edu')['ecdsa-sha2-nistp256']
    tp = paramiko.Transport(('ssh.ist.psu.edu', 22))
    password = os.getenv('PASSWORD')
    tp.connect(username='txe5135', password=password, hostkey=fingerprint)
    client = paramiko.SFTPClient.from_transport(tp)
    run_names = client.listdir('/home/txe5135/logs')
    print(run_names)
    for run_name in (pbar := tqdm(run_names)):
        if not os.path.exists(f'./extracted_weights/{run_name}.tar'):
            if not run_name == 'hparam_search':
                timestamp, budget, city, model_name = run_name.split('-')
                if city == 'EMA':
                    weights = client.listdir(f'/home/txe5135/logs/{run_name}/weights')
                    weight_timestamps = sorted([int(weight.split('.')[0].split('_')[1]) for weight in weights if weight != 'Attacker_final.tar'])
                    max_time = weight_timestamps[len(weight_timestamps)//2]
                    client.get(f'/home/txe5135/logs/{run_name}/weights/Attacker_{max_time}.tar', f'./extracted_weights/{run_name}.tar', callback=lambda a, b: pbar.set_description(f'{run_name} | {int(a/1024)}/{int(b/1024)}'))