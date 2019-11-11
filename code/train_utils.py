import tensorflow as tf
import random
import colorsys


def dice_score(target, output, smooth=1e-5):
    """
    Calculates the dice score for every sample in a batch
    output: [N x H x W] tensor < Scaled Logits >
    target: target seg [N x H x W] tensor
    smooth: smoothing factor
    """
    with tf.name_scope('dice_score'):
        intersection = tf.reduce_sum(tf.multiply(output, target), axis=[1, 2])
        union = tf.reduce_sum((output + target), axis=[1, 2])

        dice = 2 * (intersection + smooth) / (union + smooth)
        class_dice = tf.reduce_mean(dice, axis=0)
        total_dice = tf.reduce_mean(class_dice)
        return total_dice


def generalized_dice(target, output, smooth=1e-5):
    """
    Calculates the normalized dice score for every sample in a batch
    https://niftynet.readthedocs.io/en/dev/niftynet.layer.loss_segmentation.html
    output: [N x H x W] tensor < Scaled Logits >
    target: target seg [N x H x W] tensor
    smooth: smoothing factor
    """
    with tf.name_scope('generalized_dice_score'):
        weights = tf.reduce_sum(target, [1, 2])
        weights = tf.reciprocal(tf.square(weights))
        weights_clean = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
        # assign weights of the maximum class to absent class
        weights = tf.where(tf.is_inf(weights),
                           tf.ones_like(weights) * tf.reduce_max(weights_clean, axis=1, keep_dims=True),
                           weights)

        intersection = tf.reduce_sum(tf.multiply(output, target), [1, 2])
        union = tf.reduce_sum(output + target, [1, 2])

        dice_numerator = 2 * (tf.reduce_mean(tf.multiply(intersection, weights), axis=0))
        dice_denominator = tf.reduce_mean(tf.multiply(union, weights), axis=0)
        dice = tf.reduce_mean((dice_numerator + smooth) / (dice_denominator + smooth))

    return dice


def batch_loss(target, output, activation='softmax', loss_fn='dice'):
    """
    Calculate and return appropriate loss scores
    """
    with tf.name_scope('batch_Loss'):
        if activation == 'softmax':
            scaled_output = tf.nn.softmax(output, axis=3)
        elif activation == 'sigmoid':
            scaled_output = tf.nn.sigmoid(output)
        else:
            print('activation not understood')

        if loss_fn == 'dice':
            batch_dice_score = dice_score(target=target, output=scaled_output)
            loss = 1 - batch_dice_score
        elif loss_fn == 'cross_entropy':
            if activation == 'softmax':
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=output, axis=3)
            elif activation == 'sigmoid':
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)
        elif loss_fn == 'generalized_dice':
            gen_dice_score = generalized_dice(target=target, output=scaled_output)
            loss = 1 - gen_dice_score
        else:
            print('loss function not understood')
        return loss


def get_total_loss(target, output, activation='softmax', loss_fn='dice',
                   deep_supervision='False', aux_ops=None, aux_weight=0.5, scope='loss'):
    """
    Calculate total loss with or without deep supervision
    """
    with tf.name_scope(scope):
        loss_summary_op = []
        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regularization_loss = tf.reduce_sum(regularization_loss)
        loss_summary_op.append(tf.summary.scalar('regularization_loss', regularization_loss))

        loss = batch_loss(target=target, output=output, activation=activation, loss_fn=loss_fn)
        loss_summary_op.append(tf.summary.scalar('batch_loss', loss))

        if not deep_supervision:
            total_loss = loss + regularization_loss
            loss_summary_op.append(tf.summary.scalar('total_loss', total_loss))
        else:
            aux_losses = [batch_loss(target=target, output=aux_op, activation=activation, loss_fn=loss_fn) for aux_op in
                          aux_ops]
            aux_losses = tf.reduce_mean(aux_losses)
            loss_summary_op.append(tf.summary.scalar('aux_loss', aux_losses))
            total_loss = loss + (aux_weight * aux_losses) + regularization_loss
            loss_summary_op.append(tf.summary.scalar('total_loss', total_loss))

        return total_loss, loss_summary_op


def exponential_decay_with_warmup(global_step,
                                  learning_rate_base,
                                  learning_rate_decay_steps,
                                  learning_rate_decay_factor,
                                  warmup_steps=0,
                                  min_learning_rate=1e-6,
                                  staircase=True):
    """
    Learning rate scheduling:
    1) Constant learning rate for the warm-up period
    2) Exponential decay after warm-up period
    3) Disable learning rate from dropping below minimum value
    Adapted from: tensorflow/models/research/object_detection/utils/learning_schedules.py
    """

    exponential_learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        global_step - warmup_steps,
        learning_rate_decay_steps,
        learning_rate_decay_factor,
        staircase=staircase)

    learning_rate = tf.maximum(tf.where(tf.less(tf.cast(global_step, tf.int32), tf.constant(warmup_steps)),
                                        tf.constant(learning_rate_base),
                                        exponential_learning_rate),
                               min_learning_rate, name='learning_rate')
    return learning_rate


def random_colors(N, bright=False):
    """
    Generate list of random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    colors[0] = (0., 0., 0.)
    return colors


def display_multiclass(label_onehot, colorlist):
    """
    Return a three channel image for tensorboard display from the one hot image
    """
    assert label_onehot.shape[3] == len(colorlist), "Number of colors provided must be same as labels"
    colorlist = tf.constant(colorlist)

    all_labels = []
    for i in range(label_onehot.shape[3]):
        current_label = label_onehot[:, :, :, i]
        color = colorlist[i][tf.newaxis, tf.newaxis, tf.newaxis, :]
        current_label = tf.stack([current_label, current_label, current_label], axis=3)
        current_label = current_label * color
        all_labels.append(current_label)
    return tf.add_n(all_labels)


def eval_summaries(target, output, label_map, scope='eval'):
    """
    Add class-wise dice evaluation using onehot targets and labels
    """
    scaled_output = tf.nn.softmax(output, axis=3)
    with tf.name_scope(scope):
        summary_op = []
        dice_list = []
        n_classes = len(label_map)
        for current_class in range(n_classes):
            current_dice = dice_score(target[:, :, :, current_class], scaled_output[:, :, :, current_class])
            dice_list.append(current_dice)
            summary_op.append(tf.summary.scalar(label_map[str(current_class)], current_dice))
        dice_no_bg = tf.add_n(dice_list[1:])/(n_classes-1)
        summary_op.append(tf.summary.scalar('average_dice_no_background', dice_no_bg))
        return dice_no_bg, summary_op


def assign_to_device(device, ps_device='cpu:0'):
    """Returns a function to place variables on the ps_device.
    Code from tensorflow Issue #9517

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on.  Example values are GPU:0 and
        CPU:0.

    If ps_device is not set then the variables will be placed on the device.
    The best device for shared varibles depends on the platform as well as the
    model.  Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.

    """
    PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable', 'MutableHashTableOfTensors',
              'MutableDenseHashTable']

    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign
