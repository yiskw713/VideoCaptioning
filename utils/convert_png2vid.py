import argparse
import cv2
import glob
import os
import shutil
import tqdm


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='train a network for action recognition')
    parser.add_argument(
        'cam_dir', type=str, help='path to a directory where cams are saved.')

    return parser.parse_args()


def main():
    args = get_arguments()

    video_paths = glob.glob(
        os.path.join(args.cam_dir, "video*"))

    mp4_list = glob.glob(os.path.join(args.cam_dir, "*.mp4"))
    for mp4 in mp4_list:
        video_paths.remove(mp4)

    for video_path in tqdm.tqdm(video_paths, total=len(video_paths)):

        frame_paths = glob.glob(
            os.path.join(video_path, "*.png")
        )
        frame_paths = sorted(frame_paths)

        img = cv2.imread(frame_paths[0])
        h, w = img.shape[:2]

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(video_path + '.mp4', fourcc, 15.0, (w, h))
        video.write(img)

        for frame_path in frame_paths[1:]:
            img = cv2.imread(frame_path)
            video.write(img)

        video.release()

        shutil.rmtree(video_path)

    print("Done.")


if __name__ == "__main__":
    main()
