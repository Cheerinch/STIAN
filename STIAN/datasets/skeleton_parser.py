import os


def parse_skeleton_file(skeleton_path):

    with open(skeleton_path, 'r') as f:
        lines = f.readlines()

    frame_data = []
    idx = 0

    total_frames = int(lines[idx].strip())
    idx += 1

    for _ in range(total_frames):

        num_people = int(lines[idx].strip())
        idx += 1

        joints_2d = []

        for _ in range(num_people):

            idx += 1

            num_joints = int(lines[idx].strip())
            idx += 1

            joints = []

            for _ in range(num_joints):

                parts = lines[idx].strip().split()

                x = float(parts[0])
                y = float(parts[1])

                joints.append((x, y))

                idx += 1

            if not joints_2d:
                joints_2d = joints

        if len(joints_2d) < 17:
            joints_2d.extend([(0.0, 0.0)] * (17 - len(joints_2d)))

        elif len(joints_2d) > 17:
            joints_2d = joints_2d[:17]

        frame_data.append(joints_2d)

    return frame_data


def get_label_from_filename(filename):

    basename = os.path.basename(filename).lower()

    if 'sit' in basename:
        return 0

    elif 'walk' in basename:
        return 1

    else:
        raise ValueError(f"无法解析标签: {filename}")