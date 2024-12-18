
image_noise_base = 20
gal_seed_base = 10

# 90 degree rotation of the scene
num_rot = 2


def get_noise_seed(*, seed, noiseId, rotId):
    return image_noise_base * (seed * num_rot + rotId) + noiseId
