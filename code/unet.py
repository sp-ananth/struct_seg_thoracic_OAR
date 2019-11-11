import tensorflow as tf
slim = tf.contrib.slim


class UNet(object):

    def __init__(self, n_classes=1, reuse=False,
                 is_training=True, config_residual=True, config_supervision=True):

        self.n_classes = n_classes
        self.reuse = reuse
        self.is_training = is_training
        self.config_residual = config_residual
        self.config_supervision = config_supervision

    def _contract(self, net_out, units, scope, keep_prob=1):
        """
        Build generic down-sampling block of the U-Net architecture.
        Consists of a down sampling layer / down convolution layer followed by conv layers.
        """
        with tf.variable_scope(scope):
            # Down-sampling step
            net_out = slim.conv2d(net_out, num_outputs=units, kernel_size=2, stride=2, activation_fn=tf.nn.relu,
                                  padding='VALID', scope='conv_reduce')

            # Prepare residual
            net_reduced = net_out
            net_reduced = slim.conv2d(net_reduced, num_outputs=units, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                      padding='SAME', scope='conv_B_1')

            # Intermediate conv layers
            net_out = slim.conv2d(net_out, num_outputs=units, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                  padding='SAME', scope='conv_A_1')
            net_out = slim.conv2d(net_out, num_outputs=units, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                  padding='SAME', scope='conv_A_2')
            net_out = slim.dropout(net_out, keep_prob=keep_prob, scope='dropout')

            # Apply residual
            if self.config_residual:
                net_out += net_reduced

            return net_out

    def _expand(self, net_out, connection, units, scope, keep_prob=0.8):
        """
        Build Generic up-sampling block of the U-Net architecture.
        Consists of a up-sampling layer / transpose convolution layer followed by conv layers.
        """
        with tf.variable_scope(scope):
            # Up-sampling step
            net_out = slim.conv2d_transpose(net_out, num_outputs=units, kernel_size=2, stride=2,
                                            activation_fn=tf.nn.relu, padding='VALID', scope='conv_reduce')

            # Concat with skip connection [conv_op_crop, net] axis=W in NHWC (3)
            assert (net_out.shape[3] == connection.shape[3])
            net_out = self._crop_and_concat(connection, net_out)

            # Prepare residual
            net_reduced = net_out
            net_reduced = slim.conv2d(net_reduced, num_outputs=units, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                      padding='SAME', scope='conv_B_1')

            # Intermediate conv layers
            net_out = slim.conv2d(net_out, num_outputs=units, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                  padding='SAME', scope='conv_A_1')
            net_out = slim.conv2d(net_out, num_outputs=units, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                  padding='SAME', scope='conv_A_2')
            net_out = slim.dropout(net_out, keep_prob=keep_prob, scope='dropout')

            # Apply residual
            if self.config_residual:
                net_out += net_reduced

            # Calculate level output for Deep supervision
            aux_op = slim.conv2d(net_out, num_outputs=self.n_classes, kernel_size=1, stride=1, activation_fn=None,
                                 padding='SAME', scope='conv_end')

            return net_out, aux_op

    def _crop_and_concat(self, x1, x2):
        """
        Crop and concatenate skip connections from down-sampling layers
        """
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)

        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        x1_crop.set_shape(x2.shape)

        # Concatenate along the channel axes
        concat_output = tf.concat([x1_crop, x2], 3)

        return concat_output

    def _input_layer(self, net_out, units, scope='Input', keep_prob=1):
        """
        Input layer to have consistent skip connections to last _unet_up layer
        """
        with tf.variable_scope(scope):
            # Intermediate conv layers
            net_out = slim.conv2d(net_out, num_outputs=units, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                  padding='SAME', scope='conv_A_1')
            net_out = slim.conv2d(net_out, num_outputs=units, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                  padding='SAME', scope='conv_A_2')
            net_out = slim.dropout(net_out, keep_prob=keep_prob, scope='dropout')

            return net_out

    def _output_layer(self, net_out, scope='Output', keep_prob=0.65):
        """
        Output layer to predict final logits
        """
        with tf.variable_scope(scope):
            net_out = slim.conv2d(net_out, num_outputs=self.n_classes, kernel_size=1, stride=1, activation_fn=None,
                                  padding='SAME', scope='conv_end')
            net_out = slim.dropout(net_out, keep_prob=keep_prob, scope='dropout')
            return net_out

    def _build_network(self, batch_input):
        """
        Build the down-sampling and up-sampling layers for of U-Net
        """

        with tf.variable_scope('UNet', reuse=self.reuse):
            batch_norm_params = {'decay': 0.9, 'epsilon': 0.001}
            weight_decay = 0.00005

            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                biases_regularizer=slim.l2_regularizer(weight_decay)):
                with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
                    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=self.is_training):
                        input_features = self._input_layer(batch_input, units=16, scope='input_layer')
                        down_1 = self._contract(input_features, units=32, scope='down_block_1')
                        down_2 = self._contract(down_1, units=64, scope='down_block_2')
                        down_3 = self._contract(down_2, units=128, scope='down_block_3')
                        down_vertex = self._contract(down_3, units=256, scope='down_block_4')

                        up_3, aux_op_3 = self._expand(down_vertex, connection=down_3, units=128, scope='up_block_4')
                        up_2, aux_op_2 = self._expand(up_3, connection=down_2, units=64, scope='up_block_3')
                        up_1, aux_op_1 = self._expand(up_2, connection=down_1, units=32, scope='up_block_2')
                        output_features, aux_op_0 = self._expand(up_1, connection=input_features, units=16,
                                                                 scope='up_block_1')

                        self.final_logits = self._output_layer(output_features, scope='output_layer')

                        if self.config_supervision:
                            with tf.name_scope('combined_prediction'):
                                all_ops = [self.final_logits, aux_op_3, aux_op_2, aux_op_1, aux_op_0]
                                all_ops = [tf.image.resize_nearest_neighbor(aux_op, [batch_input.shape[1],
                                                                                     batch_input.shape[2]]) for aux_op
                                           in all_ops]
                                # Get a weighted sum of all the side outputs
                                prediction = tf.stack(all_ops, axis=4)
                                prediction = slim.conv3d(prediction, num_outputs=1, kernel_size=1, stride=1,
                                                         activation_fn=None, padding='SAME', scope='conv_end')
                                self.combined_logits = tf.squeeze(prediction, axis=4)
                                print('prediction: ', self.final_logits.shape)
                            self.aux_ops = all_ops[1:]

    def get_prediction(self, batch_input):

        self._build_network(batch_input)
        if self.config_supervision:
            return self.combined_logits, self.aux_ops
        else:
            return self.final_logits, None
