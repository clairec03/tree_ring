# registry.py

BASELINE_METHODS = {
    "watermarking": {
        "BaseWatermarkedDiffusionPipeline": "methods.watermarked_diffusion_pipeline.BaseWatermarkedDiffusionPipeline",
        "OutputPixelWatermarking": "methods.output_pixel_watermarking.OutputPixelWatermarking",
        # "TreeRingWatermark": "methods.watermarked_diffusion_pipeline.TreeRingWatermark", 
        "SmolOutputPixelWatermarking": "methods.smol_output_pixel_watermark.SmolOutputPixelWatermarking", 
        "Distribution": "methods.distribution.DistributionPipeline",
        "CIN": "methods.cin_blue.CINBlue",
    },
    "attacks": {
        "NoAttack": "attacks.base_attack.NoAttack",
        # Add other attack methods here
        "DistortionAttack": "attacks.distortion_attack.DistortionAttack",
        "ChainedDistortionAttack": "attacks.distortion_attack.ChainedDistortionAttack",
    },
}

BASELINE_TEAMS = {
    "NoWatermarkTeam": {
        "type": "blue",
        "watermark_method": "BaseWatermarkedDiffusionPipeline",
    },
    "BaseBlueTeam": {"type": "blue", "watermark_method": "OutputPixelWatermarking"},
    "NoAttackTeam": {"type": "red", "attack_method": "NoAttack"},
    # Add other baseline teams here
}

STUDENT_TEAMS = {
    "OutputPixelTeam": {"type": "blue", "watermark_method": "OutputPixelWatermarking"},
    # "TreeRing": {"type": "blue", "watermark_method": "TreeRingWatermark"},
    "Distribution": {"type": "blue", "watermark_method": "Distribution"},
    "CIN": {"type": "blue", "watermark_method": "CIN"},
    "DistortionTeam": {
        "type": "red",
        "attack_method": "DistortionAttack",
        "distortion_type": "noise",
    },
    "ChainDistortionTeam": {
        "type": "red",
        "attack_method": "ChainedDistortionAttack",
        "distortion_types": ["noise", "compression"],
    },
    "BasicStegoWatermarking": {"type": "blue", "watermark_method": "SmolOutputPixelWatermarking"},
}
