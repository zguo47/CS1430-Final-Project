import torch

def loss_with_metrics(img_ab_out, img_ab_true, name=""):
    # Loss is mean square erros
    # cost = tf.reduce_mean(tf.squared_difference(img_ab_out, img_ab_true), name="mse")
    # cost = torch.mean(torch.square(img_ab_out - img_ab_true), dim=(1, 2, 3))
    loss = torch.nn.MSELoss()
    # Metrics for tensorboard
    # summary = tf.summary.scalar("cost " + name, cost)
    return loss(img_ab_out, img_ab_true)

def training_loop(col, optimizer, dataloader, device):
    # Set up training (input queues, graph, optimizer)
    # irr = LabImageRecordReader("lab_images_*.tfrecord", dir_tfrecord)
    # read_batched_examples = irr.read_batch(batch_size, shuffle=True)
    # read_batched_examples = irr.read_one()
    loss_list = []
    col.train()
    for batch_idx, img in enumerate(dataloader):
      imgs_l, imgs_true_ab = img
      imgs_l = imgs_l.to(device)
      imgs_true_ab = imgs_true_ab.to(device)
      optimizer.zero_grad()
      imgs_ab = col.build(imgs_l) 
      batch_loss = loss_with_metrics(imgs_ab, imgs_true_ab, "training")
      if batch_idx % 10 == 0:
        print(batch_loss)
      batch_loss.backward()
      optimizer.step()
      loss_list.append(batch_loss.detach().cpu().numpy())
      optimizer.step()
    return loss_list
 


def evaluation_loop(col, dataloader, device):
    # Set up validation (input queues, graph)
    # irr = LabImageRecordReader("val_lab_images_*.tfrecord", dir_tfrecord)
    # read_batched_examples = irr.read_batch(number_of_images, shuffle=False)
    loss_list = []
    col.eval()
    for batch_idx, img in enumerate(dataloader):
      imgs_l_val, imgs_true_ab_val = img
      imgs_l_val = imgs_l_val.to(device)
      imgs_true_ab_val = imgs_true_ab_val.to(device)
      imgs_ab_val = col.build(imgs_l_val)
      batch_loss = loss_with_metrics(imgs_ab_val, imgs_true_ab_val, "validation")
      batch_loss.backward()
      loss_list.append(batch_loss.detach().cpu().numpy())
    return loss_list