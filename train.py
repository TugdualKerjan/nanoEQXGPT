import datetime
import json
import pickle
import time
import jax
import jax.numpy as jnp
import os
import equinox as eqx
import numpy as np
import optax
import tiktoken
import tensorboardX
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_path = "out/model.eqx"
eval_interval = 2000
log_interval = 1
eval_iters = 50
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False  # disabled by default
tensorboard_log = True  # disabled by default
log_project = "exp1"
log_run_name = "gpt2"  # 'run' + str(time.time())
# data
dataset = "tinystories"
gradient_accumulation_steps = 1  # used to simulate larger batch sizes
batch_size = 8  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
dtype = (
    "bfloat16"
    # if jax.devices.cuda.is_available() and torch.cuda.is_bf16_supported()
    # else "float16"
    # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
)
seed = 1
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------
# TODO Implement multi device training.

os.makedirs(os.path.dirname(out_path), exist_ok=True)
print("✅ Output dir created !")

# os.environ["XLA_FLAGS"] = "--xla_gpu_enable_tf32=true"
ptdtype = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}[
    dtype
]

data_dir = os.path.join("data", dataset)


def get_batch(split: str):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(
            os.path.join(data_dir, "validation.bin"), dtype=np.uint16, mode="r"
        )

    ix = np.random.randint(len(data) - block_size, size=(batch_size,))
    x = jnp.stack([jnp.array(data[i : i + block_size]) for i in ix])
    y = jnp.stack([jnp.array(data[i + 1 : i + 1 + block_size]) for i in ix])

    return x, y


def convert_model_to_dtype(model, dtype: str):
    def convert_pytree_to_dtype(pytree, dtype):
        def _convert(leaf):
            if eqx.is_array(leaf):
                return leaf.astype(dtype)
            else:
                return leaf

        return jax.tree_util.tree_map(_convert, pytree)

    if dtype == "bfloat16":
        model = convert_pytree_to_dtype(model, jnp.bfloat16)
    elif dtype == "float16":
        model = convert_pytree_to_dtype(model, jnp.float16)
    elif dtype == "float32":
        model = convert_pytree_to_dtype(model, jnp.float32)


# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)

# TODO : Init from others, and resume for the scheduler. Also provide checkpoints.
# TODO: Implement mixed precision. https://github.com/patrick-kidger/equinox/issues/221
# TODO: model surgery if block_size < model.config.block_size:
#     model.crop_block_size(block_size)
#     model_args['block_size'] = block_size # so that the checkpoint will have the right value
# model.to(device)
# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
key = jax.random.key(seed)

if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, key=key)  # TODO Serious issue with weight initialization...
    model = eqx.nn.inference_mode(model, False)

if init_from == "resume":
    print(f"Resuming training from {out_path}")

    def load(filename):
        with open(filename, "rb") as f:
            checkpoint_params = json.loads(f.readline().decode())
            gptconf = GPTConfig(**checkpoint_params["model_args"])

            model = GPT(gptconf, key=jax.random.key(1))
            cost_matrix = model.cost_matrix

            def mod(path, x):
                if "cost_matrix" in jax.tree_util.keystr(path):
                    return None
                return x

            model = jax.tree_util.tree_map_with_path(mod, model)
            model = eqx.tree_deserialise_leaves(f, model)

            def pouet(path, x):
                if "cost_matrix" in jax.tree_util.keystr(path):
                    return cost_matrix
                return x

            model = jax.tree_util.tree_map_with_path(pouet, model, is_leaf=lambda x: x is None)
            return (
                model,
                checkpoint_params,
            )

    model, checkpoint = load(out_path)
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint["model_args"][k]

    model = eqx.nn.inference_mode(model, False)
    # iter_num = checkpoint["iter_num"]
    # best_val_loss = checkpoint["best_val_loss"]

print("✅ Model initialized !")

lr_scheduler = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=learning_rate,
    warmup_steps=warmup_iters if init_from == "scratch" else 0,
    decay_steps=lr_decay_iters - iter_num,
    end_value=min_lr,
)

optimizer = optax.inject_hyperparams(optax.adamw)(
    learning_rate=learning_rate
)  # TOODO BETA AND LR SCHED
# if grad_clip != 0.0:
#     optimizer = optax.chain(optax.adaptive_grad_clip(grad_clip), optimizer)

print("✅ Optimizer initialized !")


@eqx.filter_jit
def compute_loss(model, x, y, key):
    keys = jax.random.split(key, x.shape[0])
    logits = jax.vmap(model, in_axes=(0, None, 0))(x, True, keys)

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=y,
    )

    return jnp.mean(loss)


def estimate_loss(model):
    out = {}
    model = eqx.nn.inference_mode(model)  # Sets the dropout to 0
    for split in ["train", "val"]:
        losses = jnp.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            loss = compute_loss(
                model, jax.lax.stop_gradient(x), y, key=jax.random.key(1)
            )
            losses = losses.at[k].set(loss.item())
        out[split] = jnp.mean(losses)
    model = eqx.nn.inference_mode(model, False)
    return out


optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

# logging
if wandb_log:
    import wandb

    wandb.init(project=log_project, name=log_run_name, config=config)

if tensorboard_log:
    from tensorboardX import SummaryWriter

    writer = SummaryWriter(
        log_dir="./runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

print("👀 Starting run !")

x, y = get_batch("train")
t0 = time.time()
running_mfu = -1.0
for local_iter_num in range(iter_num, max_iters + 1):
    # TODO: Chec`k if this is async prefetching the next batch.
    # do a training step
    if local_iter_num % eval_interval == 0:
        losses = estimate_loss(model)
        if wandb_log:
            wandb.log(  # type: ignore
                {
                    "eval/loss": losses["val"],
                },
                step=local_iter_num,
            )
        if losses["val"] < best_val_loss or always_save_checkpoint:
            # There has to be an easier way to get the count from the hyperparameters...
            # filtering = (
            #     lambda p, v: isinstance(p[-1], optax.tree_utils.NamedTupleKey)
            #     and p[-1].tuple_name == "ScaleByAdamState"
            # )
            best_val_loss = losses["val"]
            hyperparameters = {
                # "optimizer": optimizer,
                "model_args": model_args,
                # "iter_num": iter_num,
                # "best_val_loss": best_val_loss,
                "config": config,
            }
            print(f"saving checkpoint to {out_path}")

            def save(filename, hyperparams, model):
                with open(filename, "wb") as f:
                    hyperparam_str = json.dumps(hyperparams)
                    f.write((hyperparam_str + "\n").encode())
                    eqx.tree_serialise_leaves(f, model)

            save(out_path, hyperparameters, model)

    if local_iter_num == 0 and eval_only:
        break

    accumulated_grads = None
    total_loss = 0
    # for micro_step in range(gradient_accumulation_steps):
    key, k = jax.random.split(key)
    x, y = get_batch("train")
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, x, y, k)
    if wandb_log:
        wandb.log(  # type: ignore
            {
                "train/loss": loss,
            },
            step=local_iter_num,
        )
    # print(loss)
    # total_loss += loss / gradient_accumulation_steps
    # if accumulated_grads == None:
    #     accumulated_grads = grads
    # else:
    #     accumulated_grads = jax.tree.map(
    #         lambda g1, g2: g1 + g2, accumulated_grads, grads
    # )
    # accumulated_grads = jax.tree.map(
    #     lambda g: g / gradient_accumulation_steps, accumulated_grads
    # )
    updates, optimizer_state = optimizer.update(grads, optimizer_state, model)

    model = eqx.apply_updates(model, updates)
    # TODO: micro batching, gradient accumulation... prob sum the trees
    # Gradient scaling again poping up during backward pass

    # timing and logging TODO: might be an easier way to do this.
    total_loss = loss
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if local_iter_num % log_interval == 0:
        total_loss = total_loss * gradient_accumulation_steps
        if local_iter_num > 1000:
            mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        if tensorboard_log:
            writer.add_scalar("loss", total_loss, local_iter_num)
        print(
            f"iter {local_iter_num}: loss {total_loss:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        )
