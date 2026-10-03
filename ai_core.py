import os, json, pickle, glob, random, time, typing, socket
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
import tensorflow as tf

from tensorflow.keras.layers import (Dense, Dropout, Conv1D, Conv2D, MaxPooling2D, BatchNormalization,
                                     Input, Concatenate, GlobalAveragePooling1D, GlobalAveragePooling2D,
                                     Add, Layer, Subtract, Multiply, Activation, LayerNormalization, Reshape,
                                     Flatten,
                                     )
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Orthogonal

import numpy as np
from collections import deque
import static.colorful as c

# @register_keras_serializable()
# class ReduceMeanLayer(Layer):
#     def __init__(self, keepdims=True, **kwargs):
#         super().__init__(**kwargs)
#         self.keepdims = keepdims
#
#     def call(self, inputs):
#         return tf.reduce_mean(inputs, axis=1, keepdims=self.keepdims)
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], 1)
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({"keepdims": self.keepdims})
#         return config

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
    watched = 0
    episode_count = 0
    epsilon = 0.0
    st = None
    calling_socket = None
    action_count = 0
    last_loaded = 0
    last_loaded_checkpoint = ""
    started_at = 0
    custom_probs = []

    #------------HYPER-PARAMETERS------------
    gamma = 0.9 #future coef
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    checkpoint_path = r"checkpoints"
    exps_dir = r"exps"

    min_exp_threshold = 1024*1  #exp without learning
    training_buffer = deque(maxlen=1024*1024)

    batch_size = 512
    batch_new_portion = 64
    exp_save_threshold = 256 # how much exp to collect

    min_epsilon_2 = 0.05
    max_epsilon_2 = 0.75

    explode_threshold = 100
    save_interval = 5 # episodes
    collecting_buffer = deque(maxlen=exp_save_threshold * 256)
    save_count = 5    # how many checkpoints to store
    load_index = -1   # which checkpoint to load
    print_model_shape = True

    #-----------------------------------------
    personal_exp_dir = exps_dir


    def __init__(self, index, config, srv_ports, terminal):
        self.id = index
        self.mode = ["hl", "r", "cae", "ca", "pd"].index(config[index])
        group_index = config[:index].count(self.mode)
        phone_port = srv_ports[self.id] + 100
        self.st = terminal

        # if not self.GPU:
        #     tf.config.set_visible_devices([], 'GPU')
        #     gpus = tf.config.list_physical_devices('GPU')
        #     if gpus:
        #         for gpu in gpus:
        #             tf.config.experimental.set_memory_growth(gpu, True)

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

            self.edit("2", 1, c.F.color(34) +  "  ◷ learn: 0")
            self.edit("2", 2, c.F.color(34) +  "  ◷   run: 0")
            self.edit("2", 3, c.F.color(99) + f"  watched: {self.watched}")
            self.edit("2", 4, c.F.color(99) + f"   buffer: 0/{self.min_exp_threshold}")
            self.edit("2", 5, c.F.color(55) + f"   stored: 0")


            self.edit("1", 0, c.F.color(22) + "CPU")
            self.edit("1", 1, c.F.color(27) +  "    mean Q:")
            self.edit("1", 2, c.F.color(27) +  "    last Q:")
            self.edit("1", 3, c.F.color(27) +  "      loss:")
            self.edit("1", 4, c.F.color(27) + f"     batch: {self.batch_size}")
            self.edit("1", 5, c.F.color(165) + "   last score:")

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

        elif self.mode == 1:
            match group_index:
                case 0: probs = [8, 8, 2, 2, 3, 3, 3, 2]
                case 1: probs = [2, 2, 2, 1, 2, 2, 2, 2]
                case 2: probs = [2, 2, 1, 3, 6, 6, 6, 3]
                case 3: probs = [4, 4, 3, 1, 4, 4, 4, 3]
                case _: probs = [1, 1, 1, 1, 1, 1, 1, 1]

            self.custom_probs = [i/sum(probs) for i in probs]

        elif self.mode < 4:
            self.compile_models()
            self.restore_np()
            self.load_index = -2

            if self.mode == 2:
                count = config.count(self.mode)

                dif = self.max_epsilon_2 - self.min_epsilon_2
                epsilons = []
                if count == 1:
                    epsilons = [np.mean([self.min_epsilon_2, self.max_epsilon_2])]
                if count > 1:
                    increment = dif/(count-1)
                    for i in range(count):
                        epsilons.append(self.min_epsilon_2 + increment*i)

                self.epsilon = epsilons[group_index]

        elif self.mode == 4:
            self.compile_models()
            self.restore_np()

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

        b = Conv2D(32, (3, 1), padding="same")(board_i)
        b = LayerNormalization()(b)
        b = Activation("relu")(b)

        b = Conv2D(64, (3, 1), padding="same")(b)
        b = LayerNormalization()(b)
        b = Activation("relu")(b)

        b = Conv2D(64, (3, 1), padding="same")(b)
        b = LayerNormalization()(b)
        b = Activation("relu")(b)

        b = Conv2D(64, (24, 1), padding="same")(b)
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
        m = GlobalAveragePooling2D()(m)

        v = Dense(16, activation="relu")(valids_i)

        x = Concatenate()([b, t, m, v])
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)

        act_q = Dense(8, kernel_initializer=Orthogonal(gain=0.01), name="act")(x)

        crit_q = Dense(1, name="crit_d")(x)
        crit_q = Reshape((), name="crit")(crit_q)

        self.model = tf.keras.Model(
            inputs=[board_i,
                    ctype_i,
                    ntype_i,
                    htype_i,
                    move_mem_i,
                    valids_i],
            outputs= [act_q,
                      crit_q],
            name=f"A{self.id}"
        )
        if self.print_model_shape and self.mode == 0:
            for i in range(len(self.model.trainable_variables)):
                var = self.model.trainable_variables[i]
                self.roll("--",c.F.color(202) + f"{i}. {var.name}: {list_str(var.shape, True)}", )

            self.model.summary(print_fn=lambda x: self.roll("0", f"{x}"))

        self.action_model = tf.keras.models.clone_model(self.model)


    def learn(self, exp_seq):
        new_exp_seq = exp_seq.copy()

        # for i in range(len(new_exp_seq)-2, -1, -1):
        #     new_exp_seq[i]["reward"] += self.gamma * new_exp_seq[i+1]["reward"]

        if self.mode == 0:
            self.training_buffer.extend(new_exp_seq)

        self.collecting_buffer.extend(new_exp_seq)
        if len(self.collecting_buffer) >= self.exp_save_threshold:
            if self.mode > 0:
                self.save_replay()
                if self.mode == 2:
                    self.restore_np()
            if self.mode == 0:
                if len(self.training_buffer) < self.min_exp_threshold:
                    self.load_replays(True)

            self.collecting_buffer.clear()

        count = 0
        for i in os.listdir(self.exps_dir):
            folder = os.path.join(self.exps_dir, i)
            files = glob.glob(os.path.join(folder, "exp_*.pkl"))
            for file in files:
                count += int(file.split("exp_")[-1].split("_")[1].split(".pkl")[0])
        self.edit("2", 5, c.F.color(55) + f"   stored: {count}")

        if len(self.training_buffer) > self.min_exp_threshold and self.mode == 0:
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

            s, ns, values, actions, rewards, logs, overs = zip(*raw_batch)
            v = tf.constant(values, dtype=tf.float32)
            a = tf.constant(actions, dtype=tf.int8)
            r = tf.constant(rewards, dtype=tf.float32)
            l = tf.constant(logs, dtype=tf.float32)
            o = tf.constant(overs, dtype=tf.bool)

            del actions, rewards, overs, logs, values

            loss, qm, ch, r = self.tf_learn(s, ns, v, a, r, l, o)
            self.watched += self.batch_size

            del s, ns, a, r, o, l, v

            loss = loss.numpy()
            qm = qm.numpy()
            ch = ch.numpy()
            r = r.numpy()

            self.edit("1", 2, c.F.color(27) + f"    last Q: p {ch:.3f} | r {r}")

            if not (qm > self.explode_threshold or
                    qm < -self.explode_threshold):
                self.edit("1", 1, c.F.color(27) + f"    mean Q: {qm:.3f}")

            if not (loss > self.explode_threshold or
                    loss < -self.explode_threshold):
                self.edit("1", 3, c.F.color(27) + f"      loss: {loss:.5f}{c.B.reset()}")

            if (qm > self.explode_threshold or
                    qm < -self.explode_threshold):
                self.edit("1", 1, f"{c.B.color(1)}{c.F.color(0)}    mean Q: {qm:.3f}")
            if (loss > self.explode_threshold or
                    loss < -self.explode_threshold):
                self.edit("1", 3, f"{c.B.color(1)}{c.F.color(0)}      loss: {loss:.5f}{c.B.reset()}")
            self.edit("2", 4, c.F.color(99) + f"   buffer: {len(self.training_buffer)}/{self.min_exp_threshold}")
            self.edit("2", 3, c.F.color(99) + f"  watched: {self.watched}")
            
            
            # self.action_model.set_weights(self.model.get_weights())

            tafter = time.time()
            self.edit("2", 1, c.F.color(34) + f"  ◷ learn: {(tafter-tbefore):.4f}")
            self.edit("2", 2, c.F.color(34) + f"  ◷   run: {time_stamp(self.started_at)}")
            self.update()


    # TODO: решить, как сохранять опыт раздельно для каждой среды, не смешивая и не перемешивая
    @tf.function
    def tf_learn(self, states, next_states, values: tf.Tensor,
                 actions, rewards: tf.Tensor, logs: tf.Tensor,
                 overs: tf.Tensor):
        advs = tf.TensorArray(tf.float32, size=self.batch_size)
        returns = tf.TensorArray(tf.float32, size=self.batch_size)

        gae = tf.constant(0, dtype=tf.float32)

        for


        with tf.GradientTape() as tape:
            logits, vals = self.model(states, training=True)

            mask = tf.cast([tf.constant(s[5]) for s in states], tf.bool)
            logits = tf.where(mask, logits, tf.fill(tf.shape(logits), -1e8))

            actions_t = tf.random.categorical(logits, 1)[0, 0]
            log_probs = tf.nn.log_softmax(logits)
            log_probs_a = tf.gather(log_probs, actions_t, batch_dims=1)
            ratio = tf.exp(log_probs_a - logs)





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

        state_ex = tuple(tf.constant(s, dtype=tf.float32) for s in state)
        state_ex = tuple(tf.expand_dims(s, axis=0) for s in state_ex)

        logits, value = self.model(state_ex)
        mask = tf.cast(state_ex[5], tf.bool)
        logits = tf.where(mask, logits, tf.fill(tf.shape(logits), -1e8))

        action_t = tf.argmax(logits, axis=-1)[0] if self.mode == 0 else tf.random.categorical(logits, 1)[0,0]
        log_probs = tf.nn.log_softmax(logits)
        log_prob_a = tf.gather(log_probs, action_t, batch_dims=1)

        return state, action_t.numpy(), value, log_prob_a


    def new_episode(self):
        if self.mode == 0:
            self.episode_count += 1
            if self.episode_count % self.save_interval == 0:
                self.save_np()
            if self.episode_count % 2 == 0:
                np.random.shuffle(self.training_buffer)

            self.edit("2", 3, c.F.color(99) + f"  watched: {self.watched}")

            self.edit("2", 4, c.F.color(99) + f"   buffer: {len(self.training_buffer)}/{self.min_exp_threshold}")
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
                    self.training_buffer.extend(buffer)
                    count += len(buffer)

            self.last_loaded = count
            self.edit("2", 4, c.F.color(99) + f"   buffer: {len(self.training_buffer)}/{self.min_exp_threshold}")
            self.edit("2.5", 4, c.F.color(28) + f"+{count}")
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