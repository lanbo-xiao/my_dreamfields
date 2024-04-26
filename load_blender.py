import torch
import numpy as np

# z = z + t
trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

# rotate y and z Coordinate axis
rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

# rotate x and z Coordinate axis
rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180 * np.pi) @ c2w
    c2w = rot_theta(theta / 180 * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_data():
    H, W = 400, 400
    camera_angle_x = 0.6911112070083618
    # camera_angle_x means fov
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    # imgs.shape = [400,800,800,4]

    # view direction when we test
    # This is actually 40 camera poses,
    # which are used to generate a camera trajectory for synthesizing a new perspective
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]],
                               dim=0)

    # image.shape = [800,400,400,4]

    return render_poses, [H, W, focal]
