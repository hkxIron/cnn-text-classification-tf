import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filter_of_each_size, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        # 如果sequence_length不固定,变量就没办法定义
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x") # 注意:所有的batch数据长度均相同, batch*seq_length, 即固定56,并不像rnn中动态padding
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y") # batch*2
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"): # embedding及lookup一般放在cpu上
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W") # vocab_size*emb_size
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x) # [batch,seq_length,emb_size]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1) # [batch,seq_length,emb_size,1], 单通道, NHWC

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes): # 3,4,5
            with tf.name_scope("conv-maxpool-%s" % filter_size): # 注意:此处是name_scope
                # Convolution Layer, 每个宽度的filter,都有num_filters个filter
                filter_shape = [filter_size, embedding_size, 1, num_filter_of_each_size] # shape:[filter_height, filter_width, in_channels, out_channels]
                # w: [filter_size,embed, 1, num_filters]
                filter_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # b: [num_filters]
                filter_b = tf.Variable(tf.constant(0.1, shape=[num_filter_of_each_size]), name="b")
                # conv:[batch, (seq_len-filter)//stride+1, 1, num_filters], stride=1
                # =>   [batch, seq_len-filter+1, 1, num_filters]
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded, # [batch,seq_length,embed,1]
                    filter_W,
                    strides=[1, 1, 1, 1], # NHWC, 与输入数据各维度对应
                    padding="VALID", # 不填充
                    name="conv")
                # Apply nonlinearity
                # h:[batch, seq_len-filter+1, 1, num_filters]
                h = tf.nn.relu(tf.nn.bias_add(conv, filter_b), name="relu")
                # Maxpooling over the outputs
                # h:[batch, 1, 1, num_filters]
                pooled = tf.nn.max_pool( # pooled:N*1*1*num_filters
                    h, # NHWC
                    ksize=[1, sequence_length - filter_size + 1, 1, 1], # NHWC,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filter_of_each_size * len(filter_sizes) # 每个size的filter个数,以及filter size类型的个数
        # pooled_outputs: num_filter_of_each_size*[batch, 1, len(filter_sizes)]
        # h_pool: [batch_size, 1, 1, num_filters]
        self.h_pool = tf.concat(pooled_outputs, axis=3) # [batch_size, 1, 1, num_filters]
        # h_pool_flat:[batch, num_filters_total)]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            # h_drop: [batch, num_filters_total)]
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            # W: [num_filter_total,num_classes]
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            # b:[num_classes]
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W) # l2正则化只在最后output时生效
            l2_loss += tf.nn.l2_loss(b)
            # h_drop: [batch, num_filters_total)] , w:[num_filter_total,num_classes] b:[num_classes]
            # scores: [batch, num_classes]
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # pred: [batch]
            self.predictions = tf.argmax(self.scores, axis=1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
