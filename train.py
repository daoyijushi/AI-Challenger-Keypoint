import network
import reader
import util

r = reader.Reader('./data/train/', 'annotations.pkl', 128)
l_rate = 1e-3

inflow, kmaps, amaps = network.vanilla()
ref_kmap, ref_amap, k_loss, a_loss, loss = network.compute_loss(kmaps, amaps)
train_step = tf.train.AdagradOptimizer(l_rate).minimize(loss)

sess = tf.Session()

for i in range(1e8):
  tic = datetime.time()
  img, kmap, amap = r.next_batch()
  _, batch_k_loss, batch_a_loss = \
    sess.run([train_step, k_loss, a_loss], feed_dict={inflow:img, ref_kmap:kmap, ref_amap:amap})
  toc = datetime.time()
  interval = (toc - tic).microseconds()
  r.index = 0 # train on the first batch for sanity check

  if (i + 1) % 10 == 0:
    print('Iter %d, k loss %g, a loss %g, timecost %d' % \
      (i, batch_k_loss, batch_a_loss, interval))
  
  if (i + 1) % 1000 == 0:
    k, a = sess.run([kmaps,amaps], feed_dict={inflow:img[0:1]})
    img = img[0]
    k = k[0]
    a = a[0]
    util.visualization(img, k, a, str(i)+'.jpg')
