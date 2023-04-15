import socket
import dill
import time
import torch
import os
import sys
import select
import traceback

def func4(x): x * 2
def func(x):
    return x*func4(x)

def run():
    heartbeat = 0
    # Define the host and port you want to connect to
    host = os.getenv('HOST', '51.250.75.187')
    port = int(os.getenv('PORT', '65231'))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        print("\rconnecting", end="")
        sock.connect((host, port))
        print("\nconnected")
        globls = {}
        with sock.makefile('rwb') as f:
            while True:
                ready_to_read, _, _ = select.select([sock], [], [], 15)
                if not sock in ready_to_read:
                    print("No heartbeat from server. Closing connection")
                    return
            
                typ = f.read(1)
                if typ == b'0':
                    print("got new globals")
                    for k, v in dill.load(f).items():
                        globls[k] = v
                    f.write(b'0')
                    dill.dump("success", f)
                    f.flush()
                    print("success")
                elif typ == b'1':
                    print("got new task")
                    func, args, kwargs, res_typ = dill.load(f)
                    def run():
                        try:
                            res = eval("func(*args, **kwargs)", globls, {
                                "func": func,
                                "args": args,
                                "kwargs": kwargs,
                            })
                            # func(*args, **kwargs)

                            f.write(b'0')
                            if res_typ == 0:
                                torch.save(res, f)
                            elif res_typ == 1:
                                dill.dump(res, f)
                            else:
                                raise Exception("Incorrect type: " + str(res_typ))
                                
                            print("task calculated successfully")

                        except Exception as e:
                            f.write(b'1')
                            dill.dump(e, f)
                            print("exception during task: " + str(e))
                            traceback.print_exc()


                    run()
                    f.flush()
                    print("success")
                elif typ == b'h':
                    heartbeat += 1
                    f.write(b'h')
                    f.flush()
                elif typ == b'':
                    print("closed")
                    return
                else:
                    raise Exception("unknown typ "+ str(typ))

if __name__ == '__main__':
    i = 0
    while True:
        try:
            run()
        except ConnectionRefusedError:
            print("\rconnection refused. Retrying in 0.1 second" + "." * i + " " * 5, end="")
            time.sleep(0.1)
            i += 1
            i %= 5
        except ConnectionResetError:
            print("connection reset error")
            time.sleep(1)