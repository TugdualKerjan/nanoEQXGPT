# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_path = 'out-tinystories/model.eqx'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 5 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
tensorboard_log = True # override via command line if you like
log_project = "exp3"
log_run_name = "train_new_tokenzie"  # 'run' + str(time.time())

dataset = 'tinystories'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 100 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 16
n_head = 12
n_embd = 512
dropout = 0.0

learning_rate = 1e-4 # with baby networks can afford to go a bit higher
max_iters = 500
lr_decay_iters = 500 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model