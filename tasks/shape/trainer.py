import logging
from pathlib import Path

from dataset.shape_dataset.utils import extract_mesh_from_neural_field
from fit_a_nef import SignalShapeTrainer

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class CustomSignalShapeTrainer(SignalShapeTrainer):
    def extract_and_save_meshes(self, save_folder: Path):
        for i in range(self.num_signals):
            mesh = extract_mesh_from_neural_field(self.apply_model, shape_idx=i)
            export_path = save_folder / Path(f"mesh-{i}.obj")
            mesh.export(export_path)

    def verbose_train_model(self):
        for step_num in range(1, self.num_steps + 1):
            # Train model for one epoch, and log avg loss
            coords = self.ram_process_batch()
            self.state, losses = self.train_step(self.state, coords, self.batch_occ)

            if self.log_cfg is not None:
                if step_num % self.log_cfg.loss == 0 or (step_num == self.num_steps):
                    learning_rate = self.get_lr()
                    if WANDB_AVAILABLE and self.log_cfg.use_wandb:
                        wandb.log(
                            {
                                "loss": losses.mean(),
                                "lr": learning_rate,
                            },
                            step=step_num,
                        )
                    logging.info(f"Step: {step_num}. Loss: {losses.mean()}. LR {learning_rate}")

                if step_num % self.log_cfg.metrics == 0 or (step_num == self.num_steps):
                    mean_iou, mean_iou_squared = self.iou()
                    if WANDB_AVAILABLE and self.log_cfg.use_wandb:
                        wandb.log(
                            {
                                "iou": mean_iou,
                            },
                            step=step_num,
                        )
                    logging.info(f"Step: {step_num}. IOU: {mean_iou}")

                if step_num % self.log_cfg.meshes == 0 or (step_num == self.num_steps):
                    if WANDB_AVAILABLE and self.log_cfg.use_wandb:
                        for i in range(self.max_shapes_logged):
                            mesh = extract_mesh_from_neural_field(self.apply_model, shape_idx=i)
                            export_path = self.log_cfg.shapes_temp_dir / Path(f"mesh-{i}.obj")
                            mesh.export(export_path)
                            wandb.log(
                                {
                                    f"mesh-{i}": wandb.Object3D(
                                        str(export_path), parse_model_format="obj"
                                    ),
                                },
                                step=step_num,
                            )
                        else:
                            logging.info("Wandb not available. Skipping logging shapes.")
