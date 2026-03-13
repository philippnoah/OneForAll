from gp.lightning.module_template import BaseTemplate


class GraphPredLightning(BaseTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._printed_sample = False

    def forward(self, batch):
        return self.model(batch)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        score, loss = self.compute_results(
            batch, batch_idx,
            self.exp_config.test_state_name[dataloader_idx],
            log_loss=False,
        )
        if not self._printed_sample and score is not None:
            labels = batch.bin_labels[batch.true_nodes_mask]
            preds = score.view(-1)
            n = min(5, len(labels))
            print("\n--- Sample predictions (first batch) ---")
            print(f"{'idx':>4}  {'score':>8}  {'label':>6}")
            for i in range(n):
                print(f"{i:>4}  {preds[i].item():>8.4f}  {labels[i].item():>6.0f}")
            print("----------------------------------------\n")
            self._printed_sample = True
