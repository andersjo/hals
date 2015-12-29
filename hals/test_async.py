import asyncio
import types
from time import sleep

import numpy as np
import tensorflow as tf

@types.coroutine
def drousy(how_long):
    sleep(how_long)
    yield
    print("NOTHING MORE")


async def f1():
    print("BEFORE")
    await asyncio.sleep(1)
    print("AFTER")

loop = asyncio.get_event_loop()

cmds = [f1() for i in range(100)]

loop.run_until_complete(asyncio.wait(cmds))
# loop.run_until_complete(asyncio.wait([
# coro1(),
# coro2(),
# ]))






# EMB_DIM = 5
# MAX_SEQ_LEN = 10
#
# # Two input sequences
# s0 = [1, 2, 3]
# s1 = [4, 5]
#
# embeddings = tf.Variable(tf.random_uniform([10, EMB_DIM]))
# input_seq = tf.placeholder(tf.int32, [None, MAX_SEQ_LEN], name='input_seq')
# seq_of_embeddings = tf.nn.embedding_lookup(embeddings, input_seq)
# # Add a single channel
# seq_of_embeddings_with_ch = tf.expand_dims(seq_of_embeddings, -1)
# print(seq_of_embeddings_with_ch.get_shape())
#
# # Create convolutions of size 2 and 3
#
# num_filters = 3
# filt_2gram = tf.Variable(
#     tf.random_uniform([2, EMB_DIM, 1, num_filters])
# )
#
# conv_2gram = tf.nn.conv2d(seq_of_embeddings_with_ch, filt_2gram,
#                           strides=[1, 1, 1, 1],
#                           padding='VALID')
#
# # Second argument seems dependent on the actual sequence length?
# # Or rather, the max sequence length
# print(conv_2gram.get_shape())
#
# num_2gram_windows = conv_2gram.get_shape()[1].value
# pool_2gram = tf.nn.max_pool(conv_2gram,
#                             ksize=[1, num_2gram_windows, 1, 1],
#                             strides=[1, 1, 1, 1],
#                             padding='VALID'
#                             )
#
# pool_2gram_squeezed = tf.squeeze(pool_2gram, squeeze_dims=[1, 2])
#
# comb_layer = tf.concat(1, [pool_2gram_squeezed, pool_2gram_squeezed])
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#
#     batch_sequences = [s0, s1]
#     combined = np.zeros([len(batch_sequences), MAX_SEQ_LEN])
#     for i, seq in enumerate(batch_sequences):
#         combined[i, :len(seq)] = seq
#
#     # print(combined)
#     out = sess.run(comb_layer, {input_seq: combined})
#     print(out)
#     print(out.shape)
#
#     # print(out)
#     # print(out.shape)
#
#
