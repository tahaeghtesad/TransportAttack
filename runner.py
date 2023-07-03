import subprocess

if __name__ == '__main__':

    processes = []

    for epsilon in [5, 15, 30]:
        for norm in [2]:
            p = subprocess.Popen(['python', 'mp_ddpg.py', '--norm', str(norm), '--epsilon', str(epsilon)])
            processes.append(p)

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt as e:
        for p in processes:
            p.kill()
        raise e
