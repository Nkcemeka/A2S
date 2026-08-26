class MapConfig:
    """ 
        MapConfig contains mapping
        of symbolic layers to layers from
        feature LM.
    """
    def __init__(self, sym_layers: list, aud_layers: list):
        self.sym_layers = sym_layers
        self.aud_layers = aud_layers
        assert len(self.sym_layers) == len(self.aud_layers),\
            f"Number of layers are not the same!"

    def get_dict(self):
        return {
            sym_layer: (adapter_idx, audio_layer)
            for adapter_idx, (sym_layer, audio_layer) in enumerate(
                zip(self.sym_layers, self.aud_layers)
            )
        }
