import os, json, pickle, glob, random, time, typing, socket
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
import tensorflow as tf

from tensorflow.keras.layers import (Dense, Dropout, Conv1D, Conv2D, MaxPooling2D, BatchNormalization,
                                     Input, Concatenate, GlobalAveragePooling1D, GlobalAveragePooling2D,
                                     Add, Layer, Subtract, Multiply)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.saving import register_keras_serializable

import numpy as np
from collections import deque
import static.colorful as c
from static.static_terminal import StaticTerminal

@register_keras_serializable()
class ReduceMeanLayer(Layer):
    def __init__(self, keepdims=True, **kwargs):
        super().__init__(**kwargs)
        self.keepdims = keepdims

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1, keepdims=self.keepdims)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def get_config(self):
        config = super().get_config()
        config.update({"keepdims": self.keepdims})
        return config

def global_var_sum(vars):
    s = 0
    for var in vars:
        s += tf.reduce_sum(var)
    return s


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
    target_model = None
    action_model = None
    mode = 0
    step_count = 0
    episode_count = 0
    epsilon = 0.03
    st = None

    #-------------------------------------
    gamma = 0.9 #future coef

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    checkpoint_path = r"checkpoints"
    exps_dir = r"exps"
    GPU = False

    min_exp_threshold = 1024*1  #exp without learning

    training_buffer = deque(maxlen=1024*1024)

    batch_size = 256
    batch_new_portion = 128

    regularizer = l2(0.001)
    explode_threshold = 100

    #TM_start_randomness = 0
    #TM_soft_rate = 0.08      # every step

    save_interval = 5 # episodes
    exp_save_threshold = 256 # how much exp to collect
    collecting_buffer = deque(maxlen=exp_save_threshold * 256)
    save_count = 8    # how many checkpoints to store
    load_index = -1   # which checkpoint to load

    print_model_shape = True

    #-----------------------------------------
    personal_exp_dir = exps_dir
    started_at = 0
    custom_probs = []
    last_loaded_checkpoint = ""
    calling_socket = None
    action_count = 0

    def __init__(self, index, config, srv_ports, terminal):
        self.id = index
        self.mode = config[index]
        group_index = config[:index].count(self.mode)
        phone_port = srv_ports[self.id] + 100
        self.st = terminal

        if not self.GPU:
            tf.config.set_visible_devices([], 'GPU')
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

        self.personal_exp_dir = os.path.join(self.exps_dir, str(self.id))
        if not os.path.isdir(self.personal_exp_dir):
            os.makedirs(self.personal_exp_dir, exist_ok=True)

        self.gamma = tf.constant(self.gamma, dtype=tf.float32)

        if self.mode == 0:
            self.exp_save_threshold //= 2

            self.epsilon = 0.0

            if not os.path.isdir(self.checkpoint_path):
                os.makedirs(self.checkpoint_path, exist_ok=True)
            if not os.path.isdir(self.exps_dir):
                os.makedirs(self.exps_dir, exist_ok=True)

            self.st.add_rect("--", 92, 0, 155, 45, True, c.F.color(125))
            self.st.add_rect("1", 0, 0, 46, 8, True, c.F.color(57))
            self.st.add_rect("2", 46, 0, 92, 8, True, c.F.color(57))
            self.st.add_rect("2.5", 68, 1, 91, 7, False)

            self.edit("2", 1, c.F.color(99) + " episode:")
            self.edit("2", 2, c.F.color(99) + "    step:")
            self.edit("2", 3, c.F.color(99) + " watched:")
            self.edit("2", 4, c.F.color(99) + f" epsilon:{c.F.color(201)} {self.epsilon:.2f}")
            self.edit("2", 5, c.F.color(99) + "  buffer:")
            self.edit("2.5", 2, c.F.color(227) + "last result:")

            if self.GPU:
                self.edit("2.5", 6, c.F.color(84) + " " * 21 + "GPU")
            else:
                self.edit("2.5", 6, c.F.color(35) + " " * 21 + "CPU")

            self.edit("1", 1, c.F.color(45) + "    mean Q:")
            self.edit("1", 2, c.F.color(45) + "    last Q:")
            self.edit("1", 3, c.F.color(45) + "      loss:")
            self.edit("1", 4, c.F.color(35) + " learntime:")
            self.edit("1", 5, c.F.color(35) + "  worktime:")
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

        if self.mode == 1:
            match group_index:
                case 0: probs = [8, 8, 2, 2, 3, 3, 3, 2]
                case 1: probs = [2, 2, 2, 1, 2, 2, 2, 2]
                case 2: probs = [2, 2, 1, 3, 6, 6, 6, 3]
                case 3: probs = [4, 4, 3, 1, 4, 4, 4, 3]
                case _: probs = [1, 1, 1, 1, 1, 1, 1, 1]

            self.custom_probs = [i/sum(probs) for i in probs]

        if self.mode == 2:
            self.compile_models()
            self.restore_np()
            self.load_index = -2

            count = config.count(self.mode)

            min_eps = 0.05
            max_eps = 0.81
            dif = max_eps - min_eps
            epsilons = []
            if count == 1:
                epsilons = [np.mean([min_eps, max_eps])]
            if count > 1:
                increment = dif/(count-1)
                for i in range(count):
                    epsilons.append(min_eps + increment*i)

            self.epsilon = epsilons[group_index]

        if self.mode > 0:
            self.calling_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.calling_socket.bind(("127.0.0.1", phone_port))
            self.calling_socket.settimeout(120)

            self.calling_socket.connect(("127.0.0.1", srv_ports[config.index(0)]))
            self.roll("0",c.F.color(227) + f"finished init (collecting mode)")

            if self.mode == 1:
                self.calling_socket.sendall((json.dumps({"ping": 1, "epsilon": 1}) + "\n").encode())
            if self.mode == 2:
                self.calling_socket.sendall((json.dumps({"ping": 1, "epsilon": self.epsilon}) + "\n").encode())


    def compile_models(self):
        board_i = Input(shape=[24, 10, 1], name="board")
        ctype_i = Input(shape=[7, ], name="ctype")
        ntype_i = Input(shape=[7, ], name="ntype")
        htype_i = Input(shape=[7, ], name="htype")
        move_mem_i = Input(shape=[8, 9], name="move_mem")
        valids_i = Input(shape=[8, ], name="valids")

        board_n = Conv2D(32, [3, 3], padding="same",
                         activation="relu", kernel_regularizer=self.regularizer)(board_i)
        board_n = BatchNormalization()(board_n)
        board_n = Conv2D(32, [3, 3], padding="same",
                         activation="relu", kernel_regularizer=self.regularizer)(board_n)
        board_n = MaxPooling2D((2, 2))(board_n)

        board_n = Conv2D(64, [3, 3], padding="same",
                         activation="relu", kernel_regularizer=self.regularizer)(board_n)
        board_n = BatchNormalization()(board_n)
        board_n = Conv2D(64, [3, 3], padding="same",
                         activation="relu", kernel_regularizer=self.regularizer)(board_n)
        board_n = MaxPooling2D((2, 2))(board_n)

        board_n = Conv2D(128, [3, 3], padding="same",
                         activation="relu", kernel_regularizer=self.regularizer)(board_n)
        board_n = BatchNormalization()(board_n)
        board_n = GlobalAveragePooling2D()(board_n)
        board_n = Dense(256, activation="relu", kernel_regularizer=self.regularizer)(board_n)

        ctype_n = Dense(32, activation="relu", kernel_regularizer=self.regularizer)(ctype_i)

        ntype_n = Dense(32, activation="relu", kernel_regularizer=self.regularizer)(ntype_i)

        htype_n = Dense(32, activation="relu", kernel_regularizer=self.regularizer)(htype_i)

        move_mem_n = Conv1D(32, kernel_size=3, padding="same",
                            activation="relu", kernel_regularizer=self.regularizer)(move_mem_i)
        move_mem_n = Conv1D(32, kernel_size=3, padding="same",
                            activation="relu", kernel_regularizer=self.regularizer)(move_mem_n)
        move_mem_n = GlobalAveragePooling1D()(move_mem_n)
        move_mem_n = Dense(32, activation="relu", kernel_regularizer=self.regularizer)(move_mem_n)

        valids_n = Dense(16, activation="relu", kernel_regularizer=self.regularizer)(valids_i)

        combined_n = Concatenate()([
            board_n,
            ctype_n,
            ntype_n,
            htype_n,
            move_mem_n,
            valids_n,
        ])

        x = Dense(512, activation="relu", kernel_regularizer=self.regularizer)(combined_n)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        residue_n = Dense(512, activation="relu", kernel_regularizer=self.regularizer)(x)
        residue_n = BatchNormalization()(residue_n)

        x = Add()([x, residue_n])
        x = Dense(512, activation="relu", kernel_regularizer=self.regularizer)(x)
        x = Dropout(0.2)(x)

        #dueling output-adjacent layers
        value = Dense(256, activation="relu")(x)
        value = Dense(1)(value)

        acts_value = Dense(256, activation="relu")(x)
        acts_value = Dense(8)(acts_value)
        acts_value = Multiply()([valids_i, acts_value])

        act_mean = ReduceMeanLayer()(acts_value)

        centered_act_value = Subtract()([acts_value, act_mean])
        qs = Add()([value, centered_act_value])

        self.model = tf.keras.Model(
            inputs={"board": board_i,
                    "ctype": ctype_i,
                    "ntype": ntype_i,
                    "htype": htype_i,
                    "move_mem": move_mem_i,
                    "valids": valids_i},
            outputs=qs,
            name=f"A{self.id}"
        )
        if self.print_model_shape and self.mode == 0:
            for i in range(len(self.model.trainable_variables)):
                var = self.model.trainable_variables[i]
                self.roll("--",c.F.color(202) + f"{i}. {var.name}: {list_str(var.shape, True)}", )

        #self.target_model = tf.keras.models.clone_model(self.model)

        self.action_model = tf.keras.models.clone_model(self.model)

        self.episode_count = tf.Variable(0)
        self.step_count = tf.Variable(0)


    def learn(self, exp_seq):
        new_exp_seq = exp_seq.copy()

        for i in range(len(new_exp_seq)-2, -1, -1):
            new_exp_seq[i]["reward"] += self.gamma * new_exp_seq[i+1]["reward"]

        if self.mode == 0:
            self.training_buffer.extend(new_exp_seq)

        self.collecting_buffer.extend(new_exp_seq)
        if len(self.collecting_buffer) >= self.exp_save_threshold:
            if self.mode > 0:
                self.save_replay()
                if self.mode == 2:
                    self.restore_np()
            elif len(self.training_buffer) < self.min_exp_threshold:
                self.load_replays(True)
            self.collecting_buffer.clear()


        if len(self.training_buffer) > self.min_exp_threshold and self.mode == 0:
            if self.step_count % 10 == 0:
                np.random.shuffle(self.training_buffer)
            tbefore = time.time()
            if len(new_exp_seq) < self.batch_new_portion:
                raw_batch1 = []
                for i in range(self.batch_size - len(new_exp_seq)):
                    raw_batch1.append(self.training_buffer.popleft())
                raw_batch2 = new_exp_seq
                raw_batch = raw_batch1 + raw_batch2
            else:
                raw_batch1 = []
                for i in range(self.batch_size-self.batch_new_portion):
                    raw_batch1.append(self.training_buffer.popleft())
                raw_batch2 = new_exp_seq[-self.batch_new_portion:]
                raw_batch = raw_batch1 + raw_batch2
            states: dict[str, typing.Any] = {"board": [], "ctype": [], "ntype": [], "htype": [], "move_mem": [], "valids": []}
            next_states: dict[str, typing.Any] = {"board": [], "ctype": [], "ntype": [], "htype": [], "move_mem": [], "valids": []}
            actions = []
            rewards = []
            overs = []
            for exp in raw_batch:
                for key in states.keys():
                    states[key].append(exp["state"][key])
                    next_states[key].append(exp["next_state"][key])
                actions.append(exp["action"])
                rewards.append(exp["reward"])
                overs.append(exp["over"])

            s_n = {k: np.array(v) for k, v in states.items()}
            ns_n = {k: np.array(v) for k, v in next_states.items()}
            a_n = np.array(actions)
            r_n = np.array(rewards)
            o_n = np.array(overs)

            del states, next_states, actions, rewards, overs

            loss, qm, ch, r = self.tf_learn(s_n, ns_n, a_n, r_n, o_n)

            del s_n, ns_n, a_n, r_n, o_n

            loss = loss.numpy()
            qm = qm.numpy()
            ch = ch.numpy()
            r = r.numpy()

            self.edit("1", 2, c.F.color(45) + f"    last Q: p {ch:.3f} | r {r}")

            if not (qm > self.explode_threshold or
                    qm < -self.explode_threshold):
                self.edit("1", 1, c.F.color(45) + f"    mean Q: {qm:.3f}")

            if not (loss > self.explode_threshold or
                    loss < -self.explode_threshold):
                self.edit("1", 3, c.F.color(45) + f"      loss: {loss:.5f}{c.B.reset()}")

            if (qm > self.explode_threshold or
                    qm < -self.explode_threshold):
                self.edit("1", 1, f"{c.B.color(1)}{c.F.color(0)}    mean Q: {qm:.3f}")
            if (loss > self.explode_threshold or
                    loss < -self.explode_threshold):
                self.edit("1", 3, f"{c.B.color(1)}{c.F.color(0)}      loss: {loss:.5f}{c.B.reset()}")
            self.edit("2", 5, c.F.color(99) + f"  buffer: {len(self.training_buffer)}/{self.min_exp_threshold}")
            self.edit("2", 2, c.F.color(99) + f"    step: {self.step_count // 1}")
            self.edit("2", 3, c.F.color(99) + f" watched: {self.batch_size * self.step_count}")
            
            
            self.action_model.set_weights(self.model.get_weights())

            # for target_var, model_var in zip(self.target_model.trainable_variables,
            #                                  self.model.trainable_variables):
            #     target_var.assign(self.TM_soft_rate * model_var + (1.0 - self.TM_soft_rate) * target_var)

            self.step_count.assign_add(1)

            tafter = time.time()
            self.edit("1", 4, c.F.color(35) + f" learntime: {(tafter-tbefore):.4f}")
            self.edit("1", 5, c.F.color(35) + f"  worktime: {time_stamp(self.started_at)}")
            self.update()


    @tf.function
    def tf_learn(self, states, next_states, actions, rewards, overs):
        # DDQN (Double Deep Q-learning)
        rewards = tf.cast(rewards, dtype="float32")
        # overs = tf.cast(overs, dtype="float32")
        #
        # main_qs_gamma = self.model(next_states, training=False)
        # main_best = tf.argmax(main_qs_gamma, axis=1)
        # act_indices = tf.stack([tf.range(self.batch_size, dtype=main_best.dtype), main_best], axis=1)
        # target_qs_gamma = self.target_model(next_states, training=False)
        # estimated_qs = rewards + self.gamma * (1-overs) * tf.gather_nd(target_qs_gamma, act_indices)
        # estimated_qs = tf.stop_gradient(estimated_qs)


        with tf.GradientTape() as tape:
            action_mask = tf.one_hot(actions, 8)
            qs = self.model(states, training=True)
            chosen_action_q = tf.reduce_sum(action_mask*qs, axis=1)

            loss = tf.keras.losses.Huber()(rewards, chosen_action_q) # <- estimated
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        mean_q = tf.reduce_mean(chosen_action_q)
        # mean_est_q = tf.reduce_mean(estimated_qs)

        return loss, mean_q, chosen_action_q[-1], rewards[-1]


    def action(self, state):
        if self.mode > 0:
            self.action_count += 1
            if self.action_count >= 25:
                self.action_count = 0
                self.calling_socket.sendall((json.dumps({"ping": 1}) + "\n").encode())
        if self.mode == 1:
            return state, np.random.choice(8, p=self.custom_probs)
        elif random.random() < self.epsilon:
            return state, random.randint(0, 7)

        state_np = {
            "board": np.expand_dims(np.array(state["board"], dtype=np.float32), axis=(0, -1)),
            "ctype": np.expand_dims(np.array(state["ctype"], dtype=np.float32), axis=0),
            "ntype": np.expand_dims(np.array(state["ntype"], dtype=np.float32), axis=0),
            "htype": np.expand_dims(np.array(state["htype"], dtype=np.float32), axis=0),
            "move_mem": np.expand_dims(np.array(state["move_mem"], dtype=np.float32), axis=0),
            "valids": np.expand_dims(np.array(state["valids"], dtype=np.float32), axis=0),
        }

        qs = self.action_model(state_np).numpy()[0]

        for i in range(8):
            if not state["valids"][i]:
                qs[i] = -np.inf

        return state, int(np.argmax(qs))


    def new_episode(self):
        if self.mode == 0:
            self.episode_count.assign_add(1)
            if self.episode_count % self.save_interval == 0:
                if self.id == 0:
                    self.save_np()

            self.edit("2", 1, c.F.color(99) + f" episode: {self.episode_count // 1}")
            self.edit("2", 2, c.F.color(99) + f"    step: {self.step_count // 1}")
            self.edit("2", 3, c.F.color(99) + f" watched: {self.batch_size * self.step_count}")
            self.edit("2", 5, c.F.color(99) + f"  buffer: {len(self.training_buffer)}/{self.min_exp_threshold}")
            self.update()

        if self.mode == 1:
            common_exps = 0
            for folder in [os.path.join(self.exps_dir, i) for i in os.listdir(self.exps_dir)]:
                for file in glob.glob(os.path.join(folder, "exp_*.pkl")):
                    common_exps += int(file.split("_")[-1].split(".pkl")[0])


    def save_np(self):
        # main_weights = [w.numpy() for w in self.model.trainable_variables]
        #
        # np.savez(os.path.join(self.checkpoint_path, f"checkpoint_ep{self.episode_count.numpy()}_{self.step_count.numpy()}.npz"),
        #          *main_weights)
        # self.model.save_weights()
        self.model.save(os.path.join(self.checkpoint_path, f"ckpt_ep{self.episode_count.numpy()}_{self.step_count.numpy()}.keras"))
        self.roll("0",c.F.color(2) + f"[{time_stamp()}] saved model")
        files = sorted(glob.glob(os.path.join(self.checkpoint_path, "ckpt_ep*.keras")),
                       key = lambda x: int(x.split("_ep")[-1][:-4].split("_")[0]))

        if len(files)>self.save_count:
            for f in files[:-self.save_count]:
                os.remove(f)


    def restore_np(self):
        files = sorted(glob.glob(os.path.join(self.checkpoint_path, "ckpt_ep*.keras")))

        if not files:
            self.roll("0",c.F.color(202) + f"[{time_stamp()}] no checkpoint files")
            return

        checkpoint = files[self.load_index]
        if checkpoint != self.last_loaded_checkpoint:
            self.model = tf.keras.models.load_model(os.path.abspath(checkpoint))

            self.action_model.set_weights(self.model.get_weights())

            self.episode_count.assign(int(checkpoint.split("_ep")[-1][:-6].split("_")[0]))
            self.step_count.assign(int(checkpoint.split("_ep")[-1][:-6].split("_")[1]))
            self.roll("0",c.F.color(2) + f"[{time_stamp()}] restored")
            self.last_loaded_checkpoint = checkpoint


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
                    self.training_buffer.extend(buffer)
                    count += len(buffer)

            self.edit("2", 5, c.F.color(99) + f"  buffer: {len(self.training_buffer)}/{self.min_exp_threshold}")
            self.edit("2.5", 5, c.F.color(22) + f"+{count}")
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