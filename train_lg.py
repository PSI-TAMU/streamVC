'''
https://github.com/fishaudio/fish-speech/blob/main/fish_speech/train.py
https://huggingface.co/fishaudio/fish-agent-v0.1-3b
'''



def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--checkpoint_path', default='experiments/base/encodervq')
    parser.add_argument('-c', '--config', default='experiments/base/config.json')
    parser.add_argument('--training_epochs', default=6000, type=int)
    parser.add_argument('--training_steps', default=300000, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)

    training_filelist, validation_filelist = get_dataset_filelist(h)

    model = VQCodeLightningModule(h, training_filelist, validation_filelist)

    logger = TensorBoardLogger(save_dir=a.checkpoint_path, name="logs")
    checkpoint_callback = ModelCheckpoint(dirpath=a.checkpoint_path, monitor="val_accuracy", save_top_k=3, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        max_epochs=a.training_epochs,
        gpus=h.num_gpus,
        accelerator="ddp" if h.num_gpus > 1 else None,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor]
    )

    trainer.fit(model)
