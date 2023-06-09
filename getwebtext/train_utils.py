import os
import glob
from pytorch_lightning import Trainer
import torch
import yaml
from GPUtil import showUtilization as gpu_usage
from pytorch_lightning.callbacks import ModelCheckpoint


def read_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

accelerator, device = ("gpu", "cuda:0") if torch.cuda.is_available() else ("cpu", "cpu")
print("Use deep learning device: %s, %s." % (accelerator,device))


def load(model, PATH):
    #model = TheModelClass(*args, **kwargs)
    state_dict = torch.load(PATH, map_location=torch.device(device))
    if 'pytorch-lightning_version' in state_dict.keys():
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    return model


def save(model, PATH):
    torch.save(model.state_dict(), PATH)
    return


def save_best(cur, prev, config, model):
    def _monitor_criterion(cur, prev):
        return cur < prev
    #print("current:", cur, ", previous:", prev)
    if _monitor_criterion(cur, prev):
        print("%s is improved from %0.4f to %0.4f." % (
            config['monitor'], prev, cur))
        save(model, config['saved_model_path'] )
        print("Improved model %s Saved." % config['saved_model_path'])
        return True
    else:
        print("%s is not improved at %0.4f." % (config['monitor'], prev))
        return False

def training_pipeline(model, train_x, val_x, nepochs, resume_ckpt=False, model_name='cf_model',
                      default_ckpt_pattern="./lightning_logs/version_*", monitor="val_loss", logger_path=None):
    #torch.jit.script(model)
    ckpt_path = None
    if resume_ckpt:
        ckpt_paths = glob.glob(default_ckpt_pattern)
        versions = [int(p.split("_")[-1]) for p in ckpt_paths]
        if versions:
            ckpt_version = str(max(versions))
            ckpt_path = default_ckpt_pattern.replace('*', ckpt_version)
            ckpt_paths = glob.glob(os.path.join(ckpt_path, "checkpoints/*.ckpt"))
            ckpt_path = ckpt_paths[-1] if ckpt_paths else None

    print('torch jit script trainer')
    os.system("free -h")
    gpu_usage()
    checkpoint_callback = ModelCheckpoint(
        monitor = monitor,
        dirpath = logger_path,
        filename = ('%s-epoch{epoch:02d}-val_loss{val/loss:.2f}' % model_name),
        auto_insert_metric_name = False
    )
    trainer = Trainer(
        max_epochs=nepochs+1 if ckpt_path else nepochs,
        resume_from_checkpoint = ckpt_path,
        accelerator = accelerator, devices = 1,
        enable_checkpointing = ckpt_path is None,
        callbacks=[checkpoint_callback]
    )
    #print(model)
    #trainer.tune(model)
    #print("monitor metric before: ", trainer.callback_metrics[monitor].item())
    if train_x is None:
        trainer.validate(model, val_x)
        #print("monitor metric after: ", trainer.callback_metrics[monitor].item())
    else:
        trainer.fit(model, train_x, val_x)
    if val_x is None:
        return model, None
    elif train_x is None:
        return None, trainer.callback_metrics[monitor].item()
    else:
        return model, trainer.callback_metrics[monitor].item()