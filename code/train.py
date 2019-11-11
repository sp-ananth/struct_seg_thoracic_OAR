from unet import UNet
from data_loader import SegDataLoader
import train_utils

import pandas as pd
import json

import tensorflow as tf
flags = tf.app.flags

flags.DEFINE_string('train_config', None, 'Path to train config file')
flags.DEFINE_string('train_data_paths', None, 'CSV containing train images and labels')
flags.DEFINE_string('val_data_paths', None, 'CSV containing val images and labels')
flags.DEFINE_integer('n_class', None, 'number of classes for task')
flags.DEFINE_string('log_save_dir', None, 'Where to save the model that is to be trained')
flags.DEFINE_string('ckpt_load_dir', None, 'Where to load the pretrained ckpt')

flags.DEFINE_integer('batch_size', None, 'The number of images chosen in a batch')
flags.DEFINE_integer('image_size', None, 'Size of input image')
flags.DEFINE_string('optimizer', None, 'optimizer to use')
flags.DEFINE_float('learning_rate_base', None, 'Learning rate at begining.')
flags.DEFINE_integer('warmup_steps', None, 'Number of steps with constant learning rate')
flags.DEFINE_float('decay_rate', None, 'Decay factor.')
flags.DEFINE_integer('decay_steps', None, 'How many epoches before decaying.')

flags.DEFINE_string('loss_fn', "dice", 'CSV containing val images and labels')
flags.DEFINE_bool('deep_supervision', True, 'Loss type')
flags.DEFINE_float('aux_weight', 0.5, 'weight for auxillary loss')

flags.DEFINE_string('gpu_id', '', 'Which gpu to use')
FLAGS = flags.FLAGS

label_map = {'0': 'Background', '1': 'Lung_L', '2': 'Lung_R', '3': 'Heart',
             '4': 'Esophagus', '5': 'Trachea', '6': 'Spinal_Cord'}


def train():

    with tf.Graph().as_default():


        train_summary_op = []
        val_summary_op = []

        with tf.name_scope('data_loaders'):
            train_csv = pd.read_csv(FLAGS.train_data_paths)
            train_image_list = train_csv['data_list']
            train_mask_list = train_csv['label_list']
            train_data_loader = SegDataLoader(image_list=train_image_list, mask_list=train_mask_list, n_class=FLAGS.n_class,
                                              batch_size=FLAGS.batch_size, image_size=FLAGS.image_size,
                                              augment_flip=False)
            train_batch_images, train_batch_labels = train_data_loader.get_train_batch()
            print('train_batch_images: ', train_batch_images.shape, train_batch_labels.shape)
            val_csv = pd.read_csv(FLAGS.val_data_paths)
            val_image_list = val_csv['data_list']
            val_mask_list = val_csv['label_list']
            val_data_loader = SegDataLoader(image_list=val_image_list, mask_list=val_mask_list,  n_class=FLAGS.n_class,
                                            batch_size=FLAGS.batch_size, image_size=FLAGS.image_size,
                                            augment_flip=False)
            val_batch_images, val_batch_labels = val_data_loader.get_eval_batch()
            print('val_batch_images: ', val_batch_images.shape, val_batch_labels.shape)

        with tf.name_scope('build_network'):
            train_unet = UNet(n_classes=FLAGS.n_class, reuse=False, is_training=True, config_supervision=FLAGS.deep_supervision)
            val_unet = UNet(n_classes=FLAGS.n_class, reuse=True, is_training=False, config_supervision=FLAGS.deep_supervision)

            train_logits, train_aux_ops = train_unet.get_prediction(train_batch_images)
            val_logits, val_aux_ops = val_unet.get_prediction(val_batch_images)

            train_prediction = tf.where(tf.equal(train_logits, tf.reduce_max(train_logits, axis=3, keep_dims=True)),
                                        tf.ones_like(train_logits),
                                        tf.zeros_like(train_logits))
            val_prediction = tf.where(tf.equal(val_logits, tf.reduce_max(val_logits, axis=3, keep_dims=True)),
                                      tf.ones_like(val_logits),
                                      tf.zeros_like(val_logits))

            train_total_loss, train_loss_summary_op = train_utils.get_total_loss(target=train_batch_labels,
                                                                                 output=train_logits,
                                                                                 loss_fn=FLAGS.loss_fn,
                                                                                 activation='softmax',
                                                                                 deep_supervision=FLAGS.deep_supervision,
                                                                                 aux_ops=train_aux_ops,
                                                                                 scope='train_loss')
            val_total_loss, val_loss_summary_op = train_utils.get_total_loss(target=val_batch_labels,
                                                                             output=val_logits,
                                                                             loss_fn=FLAGS.loss_fn,
                                                                             activation='softmax',
                                                                             deep_supervision=FLAGS.deep_supervision,
                                                                             aux_ops=val_aux_ops,
                                                                             scope='val_loss')

            train_dice, train_eval_op = train_utils.eval_summaries(target=train_batch_labels, output=train_logits,
                                                                   label_map=label_map, scope='train_eval')
            val_dice, val_eval_op = train_utils.eval_summaries(target=val_batch_labels, output=val_logits,
                                                               label_map=label_map, scope='val_eval')
            train_summary_op.extend(train_loss_summary_op)
            train_summary_op.extend(train_eval_op)
            val_summary_op.extend(val_loss_summary_op)
            val_summary_op.extend(val_eval_op)

        with tf.name_scope('image_summaries'):

            print('train outputs: ', train_batch_labels.shape, train_prediction.shape)
            print('val outputs: ', val_batch_labels.shape, val_prediction.shape)
            colorlist = train_utils.random_colors(FLAGS.n_class)
            train_display_target = train_utils.display_multiclass(train_batch_labels, colorlist)
            train_display_output = train_utils.display_multiclass(train_prediction, colorlist)

            val_display_output = train_utils.display_multiclass(val_prediction, colorlist)
            val_display_target = train_utils.display_multiclass(val_batch_labels, colorlist)

            # Append all Image summaries to val summary op so that images get written less often
            val_summary_op.append(tf.summary.image('train/1_image', train_batch_images, max_outputs=1))
            val_summary_op.append(tf.summary.image('train/2_target_segmentation', train_display_target, max_outputs=1))
            val_summary_op.append(tf.summary.image('train/3_output_segmentation', train_display_output, max_outputs=1))

            val_summary_op.append(tf.summary.image('val/1_image', val_batch_images, max_outputs=1))
            val_summary_op.append(tf.summary.image('val/2_target_segmentation', val_display_target, max_outputs=1))
            val_summary_op.append(tf.summary.image('val/3_output_segmentation', val_display_output, max_outputs=1))

        with tf.name_scope('optimizer'):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            learning_rate = train_utils.exponential_decay_with_warmup(global_step=global_step,
                                                                      learning_rate_base=FLAGS.learning_rate_base,
                                                                      warmup_steps=FLAGS.warmup_steps,
                                                                      learning_rate_decay_factor=FLAGS.decay_rate,
                                                                      learning_rate_decay_steps=FLAGS.decay_steps)
            train_summary_op.append(tf.summary.scalar('Learning_rate', learning_rate))
            if FLAGS.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
            update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_op):
                train_op = optimizer.minimize(train_total_loss, global_step=global_step)

        train_summary_op = tf.summary.merge(train_summary_op)
        val_summary_op = tf.summary.merge(val_summary_op)

        with tf.name_scope('restore_variables'):
            var_saver = tf.train.Saver(keep_checkpoint_every_n_hours=4)

        with tf.Session() as sess:

            summary_writer = tf.summary.FileWriter(FLAGS.log_save_dir, graph=sess.graph)

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            if FLAGS.ckpt_load_dir is not None:
                var_saver.restore(sess, FLAGS.ckpt_load_dir)

            while True:

                [_, t_loss, t_dice, t_summary, step] = sess.run(
                    [train_op, train_total_loss, train_dice, train_summary_op, global_step])
                print("Step: {:6d} Loss: {:4.4f} Dice: {:4.4f}".format(step, t_loss, t_dice))
                summary_writer.add_summary(t_summary, global_step=step)

                if (step % 50) == 0:
                    [v_loss, v_dice, v_summary] = sess.run([val_total_loss, val_dice, val_summary_op])
                    print("\nVal Step: {:6d} Loss: {:4.4f} Dice: {:4.4f}\n".format(step, v_loss, v_dice))
                    summary_writer.add_summary(v_summary, global_step=step)

                if (step % 1000) == 0:
                    var_saver.save(sess, FLAGS.log_save_dir, global_step=step)


if __name__ == '__main__':

    with open(FLAGS.train_config, 'r') as f:
        config = json.load(f)
        for name, value in config.items():
            FLAGS.__flags[name].value = value

    print("GPU ID:", FLAGS.gpu_id)

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id

    train()
