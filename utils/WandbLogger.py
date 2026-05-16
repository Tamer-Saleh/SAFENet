import os

class WandbLogger:
    def __init__(self, opt):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb. "
                "Run `pip install wandb` to install it."
            )
        self._wandb = wandb

        # Clean up any stale run first
        if self._wandb.run is not None:
            self._wandb.finish()

        run_name = f"{opt.model_name}_{opt.dataset}_{opt.loss}_E{opt.num_epochs}"
        os.makedirs(opt.visual_root, exist_ok=True)

        self._wandb.init(
            project=opt.wandb_project,
            name=run_name,
            config=vars(opt) if hasattr(opt, '__dict__') else opt,
            dir=opt.visual_root,
            mode='offline'
        )
        print("wandb.run after init:", self._wandb.run)  # must NOT be None

        self.config = self._wandb.config
        self.checkpoint_dir = opt.save_dir

    def finish(self, model=None):
        if model is not None:
            self._wandb.unwatch(model)
        self._wandb.finish()

    def log_metrics(self, metrics, commit=True):
        self._wandb.log(metrics, commit=commit)

    def log_image(self, key_name, image_array):
        self._wandb.log({key_name: self._wandb.Image(image_array)})

    def log_images(self, key_name, list_images):
        self._wandb.log({key_name: [self._wandb.Image(img) for img in list_images]})

    def log_checkpoint(self, current_epoch, is_best=False):
        checkpoint_name = f"epoch_{current_epoch}.pth"
        gen_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        if not os.path.exists(gen_path):
            print(f"Warning: Checkpoint file not found at {gen_path}. Skipping artifact log.")
            return

        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_model", type="model"
        )
        model_artifact.add_file(gen_path)

        aliases = ["latest", f"epoch_{current_epoch}"]
        if is_best:
            aliases.append("best")

        self._wandb.log_artifact(model_artifact, aliases=aliases)
        print(f"W&B Artifact logged for checkpoint: {checkpoint_name}")
