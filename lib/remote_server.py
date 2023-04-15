import socket
import dill
import concurrent
import io
import queue
import threading
import select

# class FileAsyncResult:
#     def __init__(self, filepath):
#         self.event = threading.Event()
#         self.filepath = filepath
#         self.exception = None
        
#     def set_exception(self, exception):
#         self.exception = exception
#         self.event.set()

#     def set_result(self, sockfile):
#         with open(self.filepath) as file:
#             size = dill.load(sockfile)
#             while size > 0:
#                 cnt_read = min(size, 10 * 1024 * 1024)
#                 file.write(sockfile.read(cnt_read))
#                 size -= cnt_read
#         self.event.set()

class BasicAsyncResult:
    def __init__(self):
        self.event = threading.Event()
        self.result = None
        self.exception = None
        
    def set_exception(self, exception):
        self.exception = exception
        self.event.set()
        
    def set_result(self, sockfile):
        self.result = dill.load(sockfile)
        self.event.set()
        
class RemoteRunnerServer:
    def __init__(self):
        self.requests_queue = queue.Queue()

    def connection_handler(self, sock):
        with sock, sock.makefile('rwb') as sockfile:
            while True:
                try:
                    result, typ, rpc_request = self.requests_queue.get(timeout=1)
                except queue.Empty:
                    # Heartbeat
                    sockfile.write(b'h')
                    sockfile.flush()

                    # Wait 3 seconds
                    ready_to_read, _, _ = select.select([sock], [], [], 5)
                    if sockfile in ready_to_read:
                        print("heartbeat timeout. Closing connection")
                        return

                    res = sockfile.read(1)
                    if res != b'h':
                        print("Heartbeat failed. Closing connection")
                        return
                    continue
                # GLOBALS
                # sockfile.write(b'0')
                # dill.dump({
                #     "func4": func4
                # }, sockfile)
            
                # RPC COMMAND
                sockfile.write(typ) # b'1'
                dill.dump(rpc_request, sockfile)                          
                sockfile.flush()
                
                typ = sockfile.read(1)
                if typ == b'0':
                    result.set_result(sockfile)
                elif typ == b'1':
                    result.set_exception(dill.load(sockfile))
                elif typ == b'':
                    print("disconnected")
                    return
                else:
                    raise Exception("unknown type " + str(typ))
    
    def send_globals(self, globls):
        assert isinstance(globls, dict)
        
        result = BasicAsyncResult()
        self.requests_queue.put((result, b'0', globls))
        result.event.wait()
        return result.result     
    
    def rpc_simple(self, func, *args, **kwargs):
        assert callable(func)
                                
        result = BasicAsyncResult()
        self.requests_queue.put((result, b'1', ((func, args, kwargs, 1))))
        result.event.wait()
        if result.exception:
            raise result.exception
        return result.result
        
    # def rpc_file(self, file, func, *args, **kwargs):
    #     assert callable(func)
    #     assert isinstance(file, str)
    #     result = FileAsyncResult(file)
    #     self.requests_queue.put((result, b'1', ((func, args, kwargs, 1))))
    #     result.event.wait()
    #     if result.exception:
    #         raise result.exception
    #     return True
    
    def host_server(self):
        HOST = '0.0.0.0' 
        PORT = 65231

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            # listen for incoming connections
            s.listen()
            print(f"Server is listening on {HOST}:{PORT}")
            # wait for a client to connect
            
            with concurrent.futures.ThreadPoolExecutor(4, "ServerConnectionHandler") as executor:
                while True:
                    conn, addr = s.accept()
                    print(f"Connected by {addr}")
                    self.connection_handler(conn)
                    # executor.submit(self.handle_connection, conn)
                        # receive data from the client
    
    def run(self):
        self.thread = threading.Thread(target=self.host_server)
        self.thread.start()

def is_sendable(x):
    if callable(x):
        return True
    if str(type(x)) != "<class 'module'>":
        return False
    return 'built-in' not in str(x) 

def get_sendable_globals(globs):
    res = {}
    for k,v in globs.items():
        if k in ['exit', 'open', 'quit', 'get_ipython']: continue
        if not is_sendable(v): continue
        res[k] = v
    return res