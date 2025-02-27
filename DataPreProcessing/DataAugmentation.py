class DataAugmentation:
    def __init__(self, data):
        self.data = data

    # Factor construction
    # Augmentation 1: Horizontal mirroring (pose shift)
    # Augmentation 2: Bluring
    # Augmentation 3: Rotation (20 degrees max)
    # Augmentation 4: Brightness (decreased by at most 30%)
    # Augmentation 5: Contrast (decreased by at most 30%)
    # Augmentation 6: Stretching (horizontal and vertical, at most 30%)