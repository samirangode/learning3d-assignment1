"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import imageio


from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def render_tetrahedron(
    tetrahedron_path="data/tetrahedron.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    # vertices, faces = load_cow_mesh(tetrahedron_path)
    ######### tetrahedron
    # vertices = torch.tensor([[0.0,0.0,0.0],
    #                          [0.0,1.0,1.0],
    #                          [1.0,1.0,0.0],
    #                          [1.0,0.0,1.0]])
    # faces = torch.tensor([[0,1,2],
    #                       [0,1,3],
    #                       [0,2,3],
    #                       [1,2,3]])
    ######### cube
    vertices = torch.tensor([[0.5,0.5,0.5],
                             [0.5,0.5,-0.5],
                             [0.5,-0.5,0.5],
                             [0.5,-0.5,-0.5],
                             [-0.5,0.5,0.5],
                             [-0.5,0.5,-0.5],
                             [-0.5,-0.5,0.5],
                             [-0.5,-0.5,-0.5]])
    faces = torch.tensor(
        [
            [0, 1, 2],
            [1, 3, 2],  # left face: 0, 1
            [2, 3, 6],
            [3, 7, 6],  # bottom face: 2, 3
            [0, 2, 6],
            [0, 6, 4],  # front face: 4, 5
            [0, 5, 1],
            [0, 4, 5],  # up face: 6, 7
            [6, 7, 5],
            [6, 5, 4],  # right face: 8, 9
            [1, 7, 3],
            [1, 5, 7],  # back face: 10, 11
        ],
        dtype=torch.int64
    )
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Prepare the camera:
    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(
    #     R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
    # )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    image_list = []
    for i in range(0,360,4):
        # Prepare the camera:
        r, t = pytorch3d.renderer.cameras.look_at_view_transform(dist = 3, elev = 0.0, azim = i, degrees= True)
        print(i)

        print(r)

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=r, T=t, fov=60, device=device
        )

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        image_list.append(rend)
    
    
    ######## For generating the static mesh.
    # rend = renderer(mesh, cameras=cameras, lights=lights)
    # rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    # return rend
    return image_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tetrahedron_path", type=str, default="data/tetrahedron.obj")
    parser.add_argument("--output_path", type=str, default="images/tetrahedron_render.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    ### for static mesh
    # image = render_tetrahedron(tetrahedron_path=args.tetrahedron_path, image_size=args.image_size)
    ### for camera rotation.
    image_list = render_tetrahedron(tetrahedron_path=args.tetrahedron_path, image_size=args.image_size)
    
    my_images = image_list
    imageio.mimsave('cube_gif.gif', my_images, fps=20)

    # plt.imshow(image)
    # print("This works")
    # plt.imsave(args.output_path, image)
    # print("This works too")