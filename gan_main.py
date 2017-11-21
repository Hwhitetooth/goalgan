import gan_network as nn
import tensorflow as tf
import batch_helper as bh
import cv2


def main():
    i = 0
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    lsgan = nn.LSGAN(sess, 'lsgan')
    sess.run(tf.global_variables_initializer())
    while True:
        images1, labels = next(bh.get_batch_cifar_no_augmentation)
        images2, labels = next(bh.get_batch_cifar_no_augmentation)

        g_loss, d_loss, generated = lsgan.train_step(images1, images2)

        img = cv2.resize(255*generated[0][::-1], (400, 400), interpolation=cv2.INTER_NEAREST)
        i += 1
        print(i, g_loss, d_loss)
        if i % 10 == 0:
            cv2.imwrite('./sample.png', img)

main()