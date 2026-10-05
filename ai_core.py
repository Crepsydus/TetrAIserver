import os, json, pickle, glob, random, time, socket
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
import tensorflow as tf

from tensorflow.keras.layers import (Dense, Conv1D, Conv2D,
                                     Input, Concatenate, GlobalAveragePooling1D,
                                     Activation, LayerNormalization, Reshape,
                                     Flatten,
                                     )

from tensorflow.keras.initializers import Orthogonal

import numpy as np
from collections import deque
import static.colorful as c


def list_str(l, paranthesis = False):
    res = "[" if paranthesis else ""
    if len(l) > 0:
        res += str(l[0])
        if len(l) > 1:
            for i in range(1, len(l)):
                res += ", "
                res += str(l[i])
    res += "]" if paranthesis else ""
    return res


def time_stamp(custom_time = None):
    if custom_time is None:
        h = str(time.localtime()[3])
        m = str(time.localtime()[4])
        s = str(time.localtime()[5])
    else:
        t = time.time() - custom_time
        h = str(time.gmtime(t)[3])
        m = str(time.gmtime(t)[4])
        s = str(time.gmtime(t)[5])
    res = ""
    for i in [h, m, s]:
        if len(i) == 1:
            res += "0"
        res += i
        res += ":"
    res = res[:8]
    return res


class Agent:
    id = 0
    model = None
    mode = 0
    watched = 0
    episode_count = 0
    st = None
    calling_socket = None
    action_count = 0
    last_loaded_checkpoint = ""
    started_at = 0
    training_buffer = deque(maxlen=1024*256)
    cutoff_colors = [c.F.color(x) for x in [89, 124, 160, 196]]

    #------------HYPER-PARAMETERS------------
    gamma = 0.94  #future move coef
    lmbd = 0.92  #future advantage coef
    critic_coef = 0.5  # weaken critic loss vs actor loss
    entropy_coef = 0.01  # penalize for going all in to one action
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    checkpoint_path = r"checkpoints"
    exps_dir = r"exps"

    min_exp_threshold = 256  #how many exp needs to be in buffer to start learning
    batch_size = 256
    exp_save_threshold = 256  # how much exp to collect

    explode_threshold = 100
    save_interval = 5 # episodes
    collecting_buffer = deque(maxlen=exp_save_threshold * 128)
    save_count = 5    # how many checkpoints to store
    load_index = -1   # which checkpoint to load
    print_model_shape = True

    #----------------------------------------
    personal_exp_dir = exps_dir


    def __init__(self, index: int, config: list, srv_ports: list, terminal):
        self.id = index
        self.mode = ["hl", "ca", "pd"].index(config[index])
        phone_port = srv_ports[self.id] + 100
        self.st = terminal

        self.personal_exp_dir = os.path.join(self.exps_dir, str(self.id))
        if not os.path.isdir(self.personal_exp_dir):
            os.makedirs(self.personal_exp_dir, exist_ok=True)

        if self.mode == 0:
            self.exp_save_threshold //= 2

            if not os.path.isdir(self.checkpoint_path):
                os.makedirs(self.checkpoint_path, exist_ok=True)
            if not os.path.isdir(self.exps_dir):
                os.makedirs(self.exps_dir, exist_ok=True)

            self.st.add_rect("--", 100, 0, 155, 45, True, c.F.color(125))
            self.st.add_rect("1", 0, 0, 38, 8, True, c.F.color(19))
            self.st.add_rect("2", 38, 0, 76, 8, True, c.F.color(19))

            self.edit("2", 1, c.F.color(93) + f"    batch: {self.batch_size}")
            self.edit("2", 2, c.F.color(93) + f"  watched: {self.watched}")
            self.edit("2", 3, c.F.color(50) + f"   buffer: {len(self.training_buffer)}")
            self.edit("2", 4, c.F.color(37) + f"   stored: 0")
            self.edit("2", 5, c.F.color(89) + f"   cutoff: 0%")


            self.edit("1", 0, c.F.color(22) + "CPU")
            self.edit("1", 1, c.F.color(70) + " last score: 0")
            self.edit("1", 2, c.F.color(27) + "       loss:")
            self.edit("1", 3, c.F.color(27) + "           :")
            self.edit("1", 4, c.F.color(70) + "    ◷ learn: 0.0")
            self.edit("1", 5, c.F.color(70) + "    ◷   run: 00:00:00")

            self.update()

            self.compile_models()

            self.restore_np()
            self.load_replays(True)

            all_folders = [os.path.join(self.exps_dir, i) for i in os.listdir(self.exps_dir)]
            for folder in [os.path.join(self.exps_dir, str(i)) for i in range(len(config))]:
                if folder in all_folders:
                    all_folders.remove(folder)
                files = glob.glob(os.path.join(folder, "exp_*.pkl.temp"))
                for file in files:
                    os.remove(file)
            for folder in all_folders:
                files = glob.glob(os.path.join(folder, "exp_*.pkl.temp"))
                for file in files:
                    os.remove(file)
                os.rmdir(folder)

            self.roll("0",
                c.F.color(227) + f"finished init, working on {self.model.weights[0].numpy().device}")
            self.started_at = time.time()

        elif self.mode > 0:
            self.compile_models()
            self.restore_np()
            if self.mode == 1:
                self.load_index = -2

            self.calling_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.calling_socket.bind(("127.0.0.1", phone_port))
            self.calling_socket.settimeout(120)

            self.calling_socket.connect(("127.0.0.1", srv_ports[0]))
            self.roll("0",c.F.color(227) + f"finished init (collecting mode)")

            self.calling_socket.sendall((json.dumps({"ping": 1}) + "\n").encode())


    def compile_models(self):
        board_i = Input(shape=[24, 10, 1], name="board")
        ctype_i = Input(shape=[7, ], name="ctype")
        ntype_i = Input(shape=[7, ], name="ntype")
        htype_i = Input(shape=[7, ], name="htype")
        move_mem_i = Input(shape=[8, 9], name="move_mem")
        valids_i = Input(shape=[8, ], name="valids")

        b = Conv2D(32, (3, 1), padding="same")(board_i)
        b = LayerNormalization()(b)
        b = Activation("relu")(b)

        b = Conv2D(64, (3, 1), padding="same")(b)
        b = LayerNormalization()(b)
        b = Activation("relu")(b)

        b = Conv2D(64, (3, 1), padding="same")(b)
        b = LayerNormalization()(b)
        b = Activation("relu")(b)

        b = Conv2D(64, (24, 1), padding="valid")(b)
        b = LayerNormalization()(b)
        b = Activation("relu")(b)
        b = Reshape((10, 64))(b)
        b = Flatten()(b)
        b = Dense(128, activation="relu")(b)


        t = Concatenate()([ctype_i, ntype_i, htype_i])
        t = Dense(32, activation="relu")(t)
        t = Dense(32, activation="relu")(t)

        m = Conv1D(64, kernel_size=3, padding="same")(move_mem_i)
        m = Conv1D(64, kernel_size=3, padding="same")(m)
        m = GlobalAveragePooling1D()(m)

        v = Dense(16, activation="relu")(valids_i)

        x = Concatenate()([b, t, m, v])
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)

        act_q = Dense(8, kernel_initializer=Orthogonal(gain=0.01), name="act")(x)

        crit_q = Dense(1, name="crit_d")(x)
        crit_q = Reshape((), name="crit")(crit_q)

        self.model = tf.keras.Model(
            inputs={"board":board_i,
                    "ctype":ctype_i,
                    "ntype":ntype_i,
                    "htype":htype_i,
                    "move_mem":move_mem_i,
                    "valids":valids_i},
            outputs= [act_q,
                      crit_q],
            name=f"A{self.id}"
        )
        if self.print_model_shape and self.mode == 0:
            for i in range(len(self.model.trainable_variables)):
                var = self.model.trainable_variables[i]
                self.roll("--",c.F.color(202) + f"{i}. {var.name}: {list_str(var.shape, True)}", )

        self.action_model = tf.keras.models.clone_model(self.model)


    def put_states_to_dict(self, states):
        boards, ctypes, ntypes, htypes, move_mems, valids = [], [], [], [], [], []
        for s in states:
            boards.append(s[0])
            ctypes.append(s[1])
            ntypes.append(s[2])
            htypes.append(s[3])
            move_mems.append(s[4])
            valids.append(s[5])
        boards = tf.stack(boards)
        ctypes = tf.stack(ctypes)
        ntypes = tf.stack(ntypes)
        htypes = tf.stack(htypes)
        move_mems = tf.stack(move_mems)
        valids = tf.stack(valids)

        return {"board": boards,
                "ctype": ctypes,
                "ntype": ntypes,
                "htype": htypes,
                "move_mem": move_mems,
                "valids": valids}


    def collect(self, collected: list):
        if self.mode != 2:
            self.collecting_buffer.extend(collected)
            if self.mode == 0:
                self.training_buffer.extend(collected)
        if len(self.collecting_buffer) >= self.exp_save_threshold:
            if self.mode == 1:
                self.save_replay()
                self.restore_np()
            elif self.mode == 0:
                if len(self.training_buffer) < self.min_exp_threshold:
                    self.load_replays(True)
            self.collecting_buffer.clear()

        if self.mode == 0:
            self.update_stored_exp_counter()
            self.edit("2", 3, c.F.color(50) + f"   buffer: {len(self.training_buffer)}")
            self.update()


    def update_stored_exp_counter(self):
        count = 0
        for i in range(1, len(self.training_buffer)):
            folder = os.path.join(self.exps_dir, str(i))
            files = glob.glob(os.path.join(folder, "exp_*.pkl"))
            for file in files:
                count += int(file.split("exp_")[-1].split("_")[1].split(".pkl")[0])

        self.edit("2", 4, c.F.color(36) + f"   stored: {count}")


    def learn(self):
        if len(self.training_buffer) > self.min_exp_threshold and self.mode == 0:
            tbefore = time.time()

            batch = []
            cutoff_exp = 0
            for _ in range(self.batch_size):
                batch.append(self.training_buffer.pop())
            while not self.training_buffer[-1][6]:
                cutoff_exp += 1
                self.training_buffer.pop()

            percent = cutoff_exp//(self.batch_size+cutoff_exp)
            color = self.cutoff_colors[percent//25]
            self.edit("2", 5, color + f"   cutoff: {percent}%")

            states, next_states, values, actions, rewards, logs, overs = zip(*batch)
            actions = tf.convert_to_tensor(actions, dtype=tf.int8)
            logs = tf.convert_to_tensor(logs, dtype=tf.float32)
            states = self.put_states_to_dict(states)
            next_states = self.put_states_to_dict(next_states)

            _, next_values = self.model(next_states, training=False)
            advs, returns = self._compute_gae(rewards, values, overs, next_values)
            loss, every_loss = self._train_step(states, actions, logs, advs, returns)
            self.watched += self.batch_size
            a_loss, c_loss, e_loss = every_loss

            del states, next_states, values, actions, rewards, logs, overs, advs, returns, next_values

            if not (loss > self.explode_threshold or
                    loss < -self.explode_threshold):
                self.edit("1", 2, c.F.color(27) + f"       loss: {loss:.5f}{c.B.reset()}")
                self.edit("1", 3, c.F.color(27) + f"           : {c.F.color(34)}{a_loss:.2f}" +
                                                  f" {c.F.color(27)}| {c_loss:.2f} | {c.F.color(92)}{e_loss:.2f}{c.B.reset()}")

            else:
                self.edit("1", 2, f"{c.B.color(1)}{c.F.color(0)}       loss: {loss:.5f}{c.B.reset()}")
                self.edit("1", 3, f"{c.B.color(1)}{c.F.color(0)}           : {a_loss:.2f} | {c_loss:.2f} | {e_loss:.2f}{c.B.reset()}")


            self.edit("2", 1, c.F.color(99) + f"  watched: {self.watched}")
            self.edit("2", 3, c.F.color(50) + f"   buffer: {len(self.training_buffer)}")

            tafter = time.time()
            self.edit("1", 4, c.F.color(70) + f"    ◷ learn: {(tafter-tbefore):.4f}")
            self.edit("1", 5, c.F.color(70) + f"    ◷   run: {time_stamp(self.started_at)}")
            self.update()


    def _compute_gae(self, rewards: tuple, values: tuple, overs: tuple, next_values: tf.Tensor):
        next_values = next_values.numpy()

        advs = np.zeros(self.batch_size, dtype=np.float32)
        gae = 0.0

        for i in reversed(range(self.batch_size)):
            delta = (rewards[i] + self.gamma * (1 - overs[i]) * next_values[i]) - values[i]  # ошибка предсказания. >0 значит действие лучше чем мы думали
            gae = delta + self.gamma * self.lmbd * (1 - overs[i]) * gae
            advs[i] = gae

        returns = advs + values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        return tf.constant(advs, dtype=tf.float32), tf.constant(returns, dtype=tf.float32)


    @tf.function
    def _train_step(self, states, actions, logs, advs, returns):
        with tf.GradientTape() as tape:
            logits, new_values = self.model(states, training=True)

            valid_mask = tf.cast(states["valids"], tf.bool)
            logits = tf.where(valid_mask, logits, tf.fill(tf.shape(logits), -1e8))

            log_probs = tf.nn.log_softmax(logits)
            probs = tf.exp(log_probs)

            act_mask = tf.one_hot(actions, 8)
            log_probs_a = tf.reduce_sum(act_mask * log_probs, axis=1)
            ratio = tf.exp(log_probs_a - logs)

            target1 = ratio * advs
            target2 = tf.clip_by_value(ratio, 0.75, 1.25) * advs
            actor_loss = -tf.reduce_mean(tf.minimum(target1, target2))

            critic_loss = tf.reduce_mean(tf.square(returns - new_values))

            entropy = -tf.reduce_sum(probs*log_probs, axis=1)
            entropy_loss = -tf.reduce_mean(entropy)

            loss = actor_loss + self.critic_coef*critic_loss + self.entropy_coef*entropy_loss

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss, [actor_loss, critic_loss, entropy_loss]


    def action(self, state):
        if self.mode > 0:
            self.action_count += 1
            if self.action_count >= 25:
                self.action_count = 0
                self.calling_socket.sendall((json.dumps({"ping": 1}) + "\n").encode())


        logits, value = self.model(self.put_states_to_dict([state]))
        mask = tf.convert_to_tensor(state[5])
        mask = tf.cast(mask, tf.bool)
        logits = tf.where(mask, logits, tf.fill(tf.shape(logits), -1e8))

        action_t = tf.argmax(logits, axis=1)[0] if self.mode != 1 else tf.random.categorical(logits, 1)[0,0]
        log_probs = tf.nn.log_softmax(logits)
        log_prob_a = tf.gather(log_probs, [action_t], batch_dims=1)

        return state, int(action_t.numpy()), value[0].numpy(), log_prob_a


    def new_episode(self):
        if self.mode == 0:
            self.episode_count += 1
            if self.episode_count % self.save_interval == 0:
                self.save_np()
            if self.episode_count % 2 == 0:
                np.random.shuffle(self.training_buffer)

            self.edit("2", 1, c.F.color(99) + f"  watched: {self.watched}")

            self.update()



    def save_np(self):
        self.model.save(os.path.join(self.checkpoint_path, f"checkpoint_{self.watched}.keras"))
        self.roll("0",c.F.color(2) + f"[{time_stamp()}] saved model")
        files = sorted(glob.glob(os.path.join(self.checkpoint_path, "checkpoint_*.keras")),
                       key = lambda x: int(x.split("_")[-1][:-6]))

        if len(files)>self.save_count:
            for f in files[:-self.save_count]:
                os.remove(f)


    def restore_np(self):
        files = sorted(glob.glob(os.path.join(self.checkpoint_path, "checkpoint_*.keras")))

        if not files:
            self.roll("0",c.F.color(202) + f"[{time_stamp()}] no checkpoint files")
            return

        if abs(self.load_index) <= len(files):
            checkpoint = files[self.load_index]
            if checkpoint != self.last_loaded_checkpoint:
                self.model = tf.keras.models.load_model(os.path.abspath(checkpoint))

                self.action_model.set_weights(self.model.get_weights())

                self.watched = int(checkpoint.split("_")[-1][:-6])

                self.roll("0",c.F.color(2) + f"[{time_stamp()}] restored")
                self.last_loaded_checkpoint = checkpoint
        else:
            self.roll("0", c.F.color(202) + f"[{time_stamp()}] less than [load index] checkpoints")


    def save_replay(self):
        file_path = os.path.join(self.personal_exp_dir, f"exp_{len(os.listdir(self.personal_exp_dir))}_{len(self.collecting_buffer)}.pkl")
        while os.path.exists(file_path):
            file_path = os.path.join(self.personal_exp_dir, f"exp_{np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))}_{len(self.collecting_buffer)}.pkl")
        temp_path = file_path + ".temp"
        with open(temp_path, "wb") as f:
            pickle.dump(list(self.collecting_buffer), f)
            f.flush()
            os.fsync(f.fileno())

        os.rename(temp_path, file_path)

        self.roll("0",c.F.color(2) + f"[{time_stamp()}] saved {len(self.collecting_buffer)} exps")


    def load_replays(self, removing = False):
        if self.mode == 0:
            count = 0
            for i in os.listdir(self.exps_dir):
                folder = os.path.join(self.exps_dir, i)
                files = glob.glob(os.path.join(folder, "exp_*.pkl"))
                for file in files:
                    with open(file, 'rb') as f:
                        buffer = pickle.load(f)
                    if removing:
                        os.remove(file)
                    self.training_buffer[i].extend(buffer)
                    count += len(buffer)
            self.update_stored_exp_counter()
            self.edit("2", 3, c.F.color(50) + f"   buffer: {len(self.training_buffer)}")
            self.roll("0", c.F.color(2) + f"[{time_stamp()}] loaded {count} exps")


    def roll(self, name, content):
        if self.st:
            self.st.roll(name, content)
            self.st.press()

    def edit(self, name, line, content):
        if self.st:
            self.st.edit(name, line, content)

    def update(self):
        if self.st:
            self.st.press()
    
    
if __name__ == "__main__":
    pass