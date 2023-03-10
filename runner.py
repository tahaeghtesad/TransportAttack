import subprocess

if __name__ == '__main__':
    for epsilon in [15, 30]:
        for norm in [1, 2]:
            subprocess.Popen(['python', '--norm', str(norm), '--epsilon', str(epsilon)])