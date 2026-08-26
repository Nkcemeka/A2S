import pytorch_lightning as pl
from notagen_peft import NotagenPeft
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs/conf", \
    config_name="config.yaml", version_base="1.4")
def main(cfg: DictConfig):
    dm = hydra.utils.instantiate(cfg.dm, aug_cfg=cfg.augmentations, 
        store_split_dir=Path(__file__).parent / f"runs/{cfg.name}",
        batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    model = NotagenPeft(feat_name=cfg.feature.feat_name, model_type=cfg.model_type, \
        use_lora_patch=cfg.use_lora_patch, use_lora_char=cfg.use_lora_char, lora_cfg=cfg.lora, \
        map_cfg=cfg.mapping, adapter_cfg=cfg.adapter, \
        pair_loss_flag=cfg.pair_loss_flag, pair_loss_margin=cfg.pair_loss_margin)

    checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            filename='icassp-{epoch:02d}-{val_loss:.4f}',
            dirpath=str(Path(__file__).parent / f"runs/{cfg.name}"),
            save_top_k=5,
            save_last=True,
            mode="min"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )

    # create trainer
    if not cfg.resume:
        trainer = pl.Trainer(
            deterministic=True,
            callbacks=[checkpoint_callback, early_stop_callback],
            num_sanity_val_steps=0,
            log_every_n_steps=10,
            max_epochs=cfg.max_epochs,
            check_val_every_n_epoch=1, 
            logger=TensorBoardLogger(name=f"icassp2027_{cfg.name}",
                save_dir=str(Path(__file__).parent / f"tboard/{cfg.name}"))
            )
    else:
        if cfg.resume_ckpt is None:
            raise RuntimeError("Please provide a valid checkpoint path to resume training!")
        trainer = pl.Trainer(
            deterministic=True,
            callbacks=[checkpoint_callback, early_stop_callback],
            num_sanity_val_steps=0,
            log_every_n_steps=10,
            max_epochs=cfg.max_epochs,
            check_val_every_n_epoch=1,
            logger=TensorBoardLogger(name=f"icassp2027_{cfg.name}", 
                save_dir=str(Path(__file__).parent / f"tboard/{cfg.name}")),
            resume_from_checkpoint=cfg.resume_ckpt)

    trainer.fit(model, dm)

if __name__ == "__main__":
    main()
