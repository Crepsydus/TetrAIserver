import subprocess as sp
import sys, socket, time, json, struct
import static.colorful as c

sim_path = r"C:\Users\user\JetBrains-projects\PycharmProjects\TetrAIserver\simulation\sim.exe"
debug_sim_path = r"C:\Users\user\JetBrains-projects\PycharmProjects\TetrAIserver\simulation\sim.console.exe"
python_program = r"C:\Users\user\JetBrains-projects\PycharmProjects\TetrAIserver\ai_server.py"
python_exec = sys.executable
HOST = "127.0.0.1"

def full_exit(exc_type, exc_value, exc_traceback):
    if exc_type:
        if issubclass(exc_type, KeyboardInterrupt):
            print(c.F.color(88) + "[L] Keyboard interrupt (Terminating all processes)")
        elif exc_type:
            print(c.F.color(160) + f"[L] Caught: {exc_value}\n{exc_traceback}")

    for s in s_sockets:
        try:
            s.sendall((json.dumps({"terminate": 1}) + "\n").encode())
            print(f"[L] sent termination {socket_ports(s)}")
        except ConnectionResetError:
            print(f"[L] cannot send termination {socket_ports(s)}")
    time.sleep(0.5)
    for s in s_sockets:
        try:
            s.close()
        except ConnectionResetError:
            print(f"[L] socket already closed {socket_ports(s)}")

    time.sleep(2)

    for id in all_processes.keys():
        sp = all_processes[id][0]
        if sp.poll() is None:
            if "g" in id:
                print("[L] closed godot")
                sp.terminate()

    sys.exit()

def socket_ports(s:socket.socket):
    return f"[{s.getsockname()[1]} <-> {s.getpeername()[1]}]"

#----------------
sims = [0,2,2,1,1] # 0 - Head, learner; 1 - random (epsilon=1, little tweaks to the chances); 2 - smart agent, uses AI to decide + epsilon
no_sim_for = []   # use for Godot-env tests (type indexes of the sim(s) in sims array)
debug_sims = False  # launch godot console with the sims (kinda useless, cus they close almost immediately after encountering errors)
selfplay = 0  # 1 = let yourself control the game (only one sim)
#----------------

sims_count = len(sims)

if sims_count >= 10:
    print(c.B.color(124) + "TOO MANY SIMULATIONS")
    quit()
if sims.count(0) > 1:
    print(c.B.color(124) + "wrong launch configuration")
    quit()
if selfplay and sims_count != 1:
    print(c.B.color(124) + "wrong launch configuration")
    quit()

all_processes = {}
s_sockets = []
s_ports = [20000+i for i in range(sims_count)]
launcher_port = 19999

si = sp.STARTUPINFO()
si.dwFlags |= sp.STARTF_USESHOWWINDOW
si.wShowWindow = 3

sync_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sync_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 2))
sync_socket.bind((HOST, launcher_port))
sync_socket.listen(sims_count)

for i in range(sims_count):
    type = sims[i]
    with open(rf"logs\stderr\{i}.txt", "w") as errf:
        if type == 0:
            process1 = sp.Popen(
                        #f"{python_exec} -u {python_program} {s_ports} 0 {selfplay} {sims} {launcher_port}",
                        [python_exec, '-u',python_program, str(s_ports), "0", str(selfplay), str(sims), str(launcher_port)],
                        creationflags=sp.CREATE_NEW_CONSOLE,
                        startupinfo=si,
                        #stderr=errf,
                    )
            print(c.F.color(84) + f"Server started on port {s_ports[i]} in mode {type}" + c.F.reset())
            all_processes[str(s_ports[i])] = (process1, True, True)

        elif type > 0:
            process2 = sp.Popen(
                [python_exec, python_program, str(s_ports), str(i), str(selfplay), str(sims), str(launcher_port)],
                creationflags=sp.CREATE_NEW_PROCESS_GROUP,
                stderr = errf
            )
            print(c.F.color(84) + f"Server started on port {s_ports[i]} in mode {type}" + c.F.reset())
            all_processes[str(s_ports[i])] = (process2, True, False)

    time.sleep(1)
    if i not in no_sim_for:
        if debug_sims:
            p = sp.Popen([debug_sim_path, f"--port={s_ports[i]}", f"--play={selfplay}"],
                         creationflags=sp.CREATE_NEW_CONSOLE,)
        else:
            p = sp.Popen([sim_path, f"--port={s_ports[i]}", f"--play={selfplay}"],
                         creationflags=sp.CREATE_NEW_PROCESS_GROUP,)
        print(c.F.color(75) + f"Simulation started on port {s_ports[i]}" + c.F.reset())
        all_processes["g" + str(s_ports[i])] = (p, False, False)
        time.sleep(1.25)

    srv_conn, addr = sync_socket.accept()
    s_sockets.append(srv_conn)

for s in s_sockets:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 1))

sync_socket.close()

sys.excepthook = full_exit

while True:
    for id in all_processes.keys():
        if all_processes[id][2] and all_processes[id][0].poll() is not None:
            print(c.F.color(160) + f"[L] Not alive: {id}")
            full_exit(0,0, 0)
