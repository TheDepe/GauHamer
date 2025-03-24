import numpy as np
import cv2


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def get_intr(param, undistort=False, legacy_calib=False):
    intr = np.eye(3)
    if legacy_calib:
        intr[0, 0] = param["K"][0]
        intr[1, 1] = param["K"][1]
        intr[0, 2] = param["K"][2]
        intr[1, 2] = param["K"][3]
        dist = np.asarray([0, 0, 0, 0])
    else:
        intr[0, 0] = param["fx_undist" if undistort else "fx"]
        intr[1, 1] = param["fy_undist" if undistort else "fy"]
        intr[0, 2] = param["cx_undist" if undistort else "cx"]
        intr[1, 2] = param["cy_undist" if undistort else "cy"]

        # TODO: Make work for arbitrary dist params in opencv
        dist = np.asarray([param["k1"], param["k2"], param["p1"], param["p2"]])

    return intr, dist


def get_extr(param, legacy_calib=False):
    """
    returns c2w
    """
    if legacy_calib:
        extr = param["extrinsics_opencv"]
    else:
        qvec = [param["qvecw"], param["qvecx"], param["qvecy"], param["qvecz"]]
        tvec = np.asarray([param["tvecx"], param["tvecy"], param["tvecz"]])
        r = qvec2rotmat(qvec)
        extr = np.vstack([np.hstack([r, tvec[:, None]]), np.array([0,0,0,1])])
    return extr


def read_params(params_path):

    params = np.loadtxt(
        params_path,
        dtype=[
            ("cam_id", int),
            ("width", int),
            ("height", int),
            ("fx", float),
            ("fy", float),
            ("cx", float),
            ("cy", float),
            ("k1", float),
            ("k2", float),
            ("p1", float),
            ("p2", float),
            ("cam_name", "<U22"),
            ("qvecw", float),
            ("qvecx", float),
            ("qvecy", float),
            ("qvecz", float),
            ("tvecx", float),
            ("tvecy", float),
            ("tvecz", float),
        ],
    )
    params = np.sort(params, order="cam_name")
    
    params_dict = {}
    for row in params:
        params_dict[row['cam_name']] = {
            field: row[field] for field in params.dtype.names if field != 'cam_name'
        }
    return params_dict


def get_undistort_params(intr, dist, img_size):
    new_intr = cv2.getOptimalNewCameraMatrix(
        intr, dist, img_size, alpha=0, centerPrincipalPoint=True
    )
    return new_intr


def undistort_image(intr, dist_intr, dist, img):
    result = cv2.undistort(img, intr, dist, None, dist_intr)
    return result

def get_scene_extent(cam_centers):
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    radius = diagonal * 1.1
    return radius


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = np.tan((fovY / 2))
    tanHalfFovX = np.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))


def get_opengl_camera_attributes(
    K, extrins, width, height, zfar=100.0, znear=0.01, resize_factor=1.0
):
    """
    extrinsics in here is c2w
    """
    K[..., :2, :] = K[..., :2, :] * resize_factor
    width = int(width * resize_factor + 0.5)
    height = int(height * resize_factor + 0.5)
    fovx = focal2fov(K[0, 0], width)
    fovy = focal2fov(K[1, 1], height)
    world_view_transform = np.transpose(extrins) # gives us w2c
    projection_matrix = np.transpose(
        getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy)
    )

    full_proj_transform = np.matmul(world_view_transform, projection_matrix)
    camera_center = np.linalg.inv(world_view_transform)[3, :3]

    out = {
        "width": width,
        "height": height,
        "fovx": fovx,
        "fovy": fovy,
        "K": K,
        "extr": extrins,
        "world_view_transform": world_view_transform,
        "projection_matrix": projection_matrix,
        "full_proj_transform": full_proj_transform,
        "camera_center": camera_center,
    }
    return out
