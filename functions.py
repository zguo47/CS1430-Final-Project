import torch

def training_loop(model, optimizer, loss_function, data, device, vgg=None):
    loss_list = []
    model.train()
    for batch_idx, img in enumerate(data):
        imgs_l, imgs_true_ab = img
        imgs_l = imgs_l.to(device)
        imgs_true_ab = imgs_true_ab.to(device)
        optimizer.zero_grad()
        if vgg is not None:
            imgs_ab = model(imgs_l, vgg)
        else:
            imgs_ab = model(imgs_l)
        batch_loss = loss_function(imgs_ab, imgs_true_ab)
        if (batch_idx+1) % 10 == 0:
            print(f"Batch {batch_idx+1}, Loss: {batch_loss:.4f}")
        batch_loss.backward()
        optimizer.step()
        loss_list.append(batch_loss.detach().cpu().numpy())
        optimizer.step()
    return loss_list

def evaluation_loop(model, loss_function, data, device, vgg=None):
    loss_list = []
    model.eval()
    for batch_idx, img in enumerate(data):
        imgs_l_val, imgs_true_ab_val = img
        imgs_l_val = imgs_l_val.to(device)
        imgs_true_ab_val = imgs_true_ab_val.to(device)
        if vgg is not None:
            imgs_ab_val = model(imgs_l_val, vgg)
        else:
            imgs_ab_val = model(imgs_l_val)
        batch_loss = loss_function(imgs_ab_val, imgs_true_ab_val)
        batch_loss.backward()
        loss_list.append(batch_loss.detach().cpu().numpy())
    return loss_list

def training_loop_cgan(model, optimizer, train_dataloader, device):
    loss_list = []
    model.train()
 
    for batch_idx, img in enumerate(train_dataloader):
      imgs_l, imgs_true_ab = img
      optimizer.zero_grad()
      imgs_l = imgs_l.to(device)
      imgs_true_ab = imgs_true_ab.to(device)
      fake_prob, true_prob, fake_color = model.forward(imgs_l, imgs_true_ab)
      batch_loss = model.loss(fake_prob, true_prob, fake_color, imgs_true_ab)
      if (batch_idx+1) % 10 == 0:
            print(f"Batch {batch_idx+1}, Loss: {batch_loss:.4f}")
      batch_loss.backward()
      optimizer.step()
      loss_list.append(batch_loss.detach().cpu().numpy())
    return loss_list

def evaluation_loop_cgan(model, dataloader, device):
    loss_list = []
    model.eval()
    for batch_idx, img in enumerate(dataloader):
      imgs_l_val, imgs_true_ab_val = img
      imgs_l_val = imgs_l_val.to(device)
      imgs_true_ab_val = imgs_true_ab_val.to(device)
      fake_prob, true_prob, fake_color = model.forward(imgs_l_val, imgs_true_ab_val)
      batch_loss = model.loss(fake_prob, true_prob, fake_color, imgs_true_ab_val)
      batch_loss.backward()
      loss_list.append(batch_loss.detach().cpu().numpy())
    return loss_list
