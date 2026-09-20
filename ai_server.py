import socket, json, time, sys, logging, struct
import static.colorful as c
from static.static_terminal import StaticTerminal
from ai_core import Agent
from numpy import clip

HOST = "127.0.0.1"
s_ports = json.loads(sys.argv[1])

srv_idx = int(sys.argv[2])
port = s_ports[srv_idx]
player_play = bool(int(sys.argv[3]))
config = json.loads(sys.argv[4])
l_port = int(sys.argv[5])

logging.basicConfig(
            filename=f"./logs/error_{srv_idx}.log",
            filemode="w",
            level=logging.ERROR,
        )

def socket_exit():
    for i in created_sockets:
        if i is not None:
            if st:
                st.roll("0", f"[{port}] closing {i}")
            i.close()
    sys.exit()

def handle_error(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        logging.error(f"{last_msg} [{time.strftime('%d.%m.%Y at %H:%M:%S')}]")
        if st:
            st.roll("0", c.F.color(88) + f"[{port}] keyboard interrupt")
            st.press()
        socket_exit()
    else:
        logging.error(f" {last_msg} [{time.strftime('%d.%m.%Y at %H:%M:%S')}]", exc_info=(exc_type, exc_value, exc_traceback))
        socket_exit()

def socket_ports(s:socket.socket, self_init: str | bool = "ignore"):
    if self_init == "ignore":
        return f"[{s.getsockname()[1]} <-> {s.getpeername()[1]}]"
    else:
        return (f"[{c.S.style(1) if self_init else ''}{s.getsockname()[1]}{c.S.reset() if self_init else ''}"
            f" {'<' if not self_init else '-'}{'>' if self_init else '-'} "
            f"{c.S.style(1) if not self_init else ''}{s.getpeername()[1]}{c.S.reset() if not self_init else ''}]")

sys.excepthook = handle_error

last_msg = ""

created_sockets = []
st = None

if __name__ == "__main__":
    if config[srv_idx] == 0:
        st = StaticTerminal(maximized=True, auto_update=False)

        st.roll("0", c.F.color(2)+f"[{port}] started")
        st.press()

    receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    receiver_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 2))
    receiver_socket.settimeout(60)
    receiver_socket.bind((HOST, port))
    backlog = len(config)+1 if config[srv_idx] == 0 else 2
    receiver_socket.listen(backlog)

    client_conn, addr = receiver_socket.accept()
    client_conn.settimeout(7.5)
    created_sockets.append(client_conn)
    if st:
        st.roll("0", c.F.color(84) + f"client connected: {socket_ports(client_conn, False)}" + c.F.reset())
        st.press()
    buffer = {client_conn: ""}

    agent = Agent(srv_idx, config, s_ports, st)
    created_sockets.append(agent.calling_socket)

    launcher_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    launcher_socket.bind((HOST, port-1000))
    launcher_socket.connect((HOST, l_port))

    created_sockets.append(launcher_socket)
    launcher_socket.setblocking(False)
    if st:
        st.roll("0", c.F.color(84) + f"launcher connected: {socket_ports(client_conn, True)}" + c.F.reset())
        st.press()
    buffer[launcher_socket] = ""


    agents_info = {}
    if config[srv_idx] == 0:
        if len(config) > 1:
            if st:
                st.add_rect("3",46,8,92, 11 + backlog - 1, True, c.F.color(57))
                st.add_rect("30", 47, 9, 61, 10 + backlog - 1, False)
                st.add_rect("31", 62, 9, 76, 10 + backlog - 1, False)
                st.add_rect("32", 77, 9, 91, 10 + backlog - 1, False)

                st.edit("30", 1, c.F.color(99) + "process")
                st.edit("31", 1, c.F.color(99) + "epsilon")
                st.edit("32", 1, c.F.color(99) + "status")
                st.press()

        for i in range(backlog-2):
            a_conn, addr = receiver_socket.accept()
            created_sockets.append(a_conn)

            if st:
                st.roll("0", c.F.color(84) + f"agent connected: {socket_ports(a_conn, False)}" + c.F.reset())
                st.press()

            a_conn.setblocking(False)
            agents_info[a_conn] = [i, 0, -1]
            buffer[a_conn] = ""
            if st:
                st.edit("30", 2+i, c.F.color(84) + str(addr[1]))

    receiver_socket.close()

    for i in created_sockets:
        if i is not None:
            i.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 1))

    client_conn.sendall((json.dumps({"action": -1, "starter": True}) + "\n").encode())

    ai_expmem = []
    await_next_msg = False
    frozen_state = []
    new_state = []
    action = -1
    move_reward = 0

    while True:
        # reading ALL sockets
        for conn in buffer.keys():
            last_msg = f"Currently reading {socket_ports(conn, 'ignore')}"
            try:
                data = conn.recv(4096)
            except socket.timeout:
                if not player_play:
                    conn.sendall((json.dumps({"action": -1, "starter": True}) + "\n").encode())
                    if st:
                        st.roll("0", c.F.color(200) + f"sent emergency starter {socket_ports(conn, True)}")
                        st.press()
                continue
            except BlockingIOError:
                continue
            except ConnectionResetError:
                buffer.pop(conn)
                created_sockets.remove(conn)
                if config[srv_idx == 0]:
                    if conn in agents_info.keys():
                        idx = agents_info[conn][0]
                        agents_info.pop(conn)
                        label = c.F.color(196) + c.S.style(1) + c.S.style(5) + "DEAD"
                        if st:
                            st.edit("32", 2 + idx, label + c.F.reset())
                            st.press()
                break
            if data:
                buffer[conn] += data.decode()


        # processing launcher-socket data
        chunks = buffer[launcher_socket].split("\n")[:-1]
        buffer[launcher_socket] = buffer[launcher_socket].split("\n")[-1]
        for chunk in chunks:
            msg = json.loads(chunk)
            if "terminate" in msg.keys():
                if st:
                    st.roll("0", c.F.color(88) + f"[{port}] terminating...")
                    st.press()
                else:
                    print(c.F.color(88) + f"[{port}] terminating...")
                socket_exit()


        # processing client data (moves)
        chunks = buffer[client_conn].split("\n")[:-1]
        buffer[client_conn] = buffer[client_conn].split("\n")[-1]

        for chunk in chunks:
            msg = json.loads(chunk)
            if "state" in msg.keys():
                result = msg["result"]

                if await_next_msg:
                    move_reward = msg["reward"]
                    frozen_state = new_state
                    if player_play:
                        action = msg["player_action"]
                new_state = msg["state"]

                if result:
                    if st:
                        st.edit("2.5", 2, c.F.color(227) + f"last result: {result}" + c.F.reset())
                        st.press()
                    ai_expmem.append({
                        "state": frozen_state,
                        "action": action,
                        "reward": clip(-800+result, -800, -100),
                        "over": True,
                        "next_state": frozen_state,
                    })
                    await_next_msg = False
                    agent.learn(ai_expmem)
                    agent.new_episode()

                    ai_expmem.clear()

                    client_conn.sendall((json.dumps({"action": -1, "starter": True}) + "\n").encode())
                else:
                    if await_next_msg:
                        ai_expmem.append({
                            "state": frozen_state,
                            "action": action,
                            "reward": move_reward,
                            "over": False,
                            "next_state": new_state,
                        })

                        if action == 3:
                            agent.learn(ai_expmem)
                            ai_expmem.clear()

                    if not player_play:
                        frozen_state, action = agent.action(new_state)
                        dump = json.dumps({"action": action, "reward": move_reward}) + "\n"
                        client_conn.sendall(dump.encode())
                    await_next_msg = True


        # processing agents' pings
        if config[srv_idx] == 0:
            for a_conn in agents_info.keys():
                idx = agents_info[a_conn][0]

                chunks = buffer[a_conn].split("\n")[:-1]
                buffer[a_conn] = buffer[a_conn].split("\n")[-1]
                for chunk in chunks:
                    msg = json.loads(chunk)
                    if "epsilon" in msg.keys():
                        if st:
                            st.edit("31", 2 + idx, c.F.color(201) + f"{msg['epsilon']:.2f}" + c.F.reset())
                    if "ping" in msg.keys():
                        agents_info[a_conn][1] = int(time.time())

                ping_dif = int(time.time())-agents_info[a_conn][1]
                status = min(ping_dif // 4, 2)
                label = [c.F.color(84) + "OK",
                         c.F.color(208) + "LONG",
                         c.F.color(196) + c.S.style(1) + c.S.style(5) + "DEAD"][status]
                if status != agents_info[a_conn][2]:
                    agents_info[a_conn][2] = status
                    if st:
                        st.edit("32", 2 + idx, label + c.F.reset())
                        st.press()