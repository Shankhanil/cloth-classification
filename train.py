from tqdm import tqdm

def finetune_setup():
    pass

def train_one_epoch(epoch_index, args, tb_writer = None):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in tqdm(enumerate(args.train_loader)):
        # Every data instance is an input + label pair
        inputs, labels = data

        inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')

        # Zero your gradients for every batch!
        args.optimizer.zero_grad()

        # print(inputs.shape)
        # Make predictions for this batch
        outputs = args.model(inputs)
        # print(outputs)
        # exit()

        # Compute the loss and its gradients
        loss = args.loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        args.optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        # print(running_loss)
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(args.dataloader) + i + 1
            # tb_writer.summary.add_scalar('Loss/train', last_loss, tb_x)
            # print('Loss/train', last_loss, tb_x)
            # running_loss = 0.
        

    return last_loss