import pickle


class SocketBrokenException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Messenger:
    def __init__(self, socket):
        self.socket = socket

    def send(self, msg):
        msg = pickle.dumps(msg)
        self.socket.sendall(len(msg).to_bytes(8, byteorder='big'))
        self.socket.sendall(msg)

    def recv(self):
        chunks = []
        msg_len = int.from_bytes(self.socket.recv(8), byteorder='big')
        bytes_recd = 0
        while bytes_recd < msg_len:
            chunk = self.socket.recv(min(msg_len - bytes_recd, 2048))
            if chunk == b'':
                raise SocketBrokenException()
            chunks.append(chunk)
            bytes_recd += len(chunk)
        return pickle.loads(b''.join(chunks))

    def close(self):
        self.socket.close()