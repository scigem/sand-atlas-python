import os
import glob
import numpy
from tqdm import tqdm
import matplotlib.image

# font = os.path.expanduser("~/Library/Fonts/Montserrat-Medium.ttf")

# debug = True
debug = False
if debug:
    silence = ""
else:
    silence = " > /dev/null 2>&1"

current_file_path = os.path.abspath(__file__)
blender_script_path = os.path.join(os.path.dirname(current_file_path), "blender_scripts", "render_mesh.py")


def make_website_video(stl_foldername, output_foldername):
    """
    Generates a video from STL files by rendering them, converting to webm format, 
    and stitching them together into a grid.

    Parameters:
    stl_foldername (str): The folder containing the STL files.
    output_foldername (str): The folder where the final video will be saved.

    Steps:
    1. Create a blank webm video if it doesn't exist.
    2. Render each STL file into a webm video using Blender and ffmpeg.
    3. Stitch the individual videos into a grid format.
    4. Concatenate the grids into a single video.
    5. Reduce the file size of the final video.
    6. Clean up intermediate files if not in debug mode.

    Note:
    - Assumes the presence of Blender and ffmpeg in the system path.
    - Uses a maximum of 72 STL files for the video.
    - Pads the grid with blank videos if the number of STL files is less than 72.
    """

    if not os.path.exists(f"{stl_foldername}/blank.webm"):
        matplotlib.image.imsave(f"{stl_foldername}/blank.png", numpy.zeros((1000, 1000, 4)))
        os.system(
            f"ffmpeg -y -loop 1 -i {stl_foldername}/blank.png -c:v libvpx-vp9 -t 2 {stl_foldername}/blank.webm"
            + silence
        )

    files = glob.glob(f"{stl_foldername}/*.stl")
    files.sort()

    start_index = 0  # Specify the index to start rendering from
    max_files = min(72, len(files) - start_index)  # Calculate max_files based on start_index
    for i, file in tqdm(enumerate(files[start_index : start_index + max_files]), total=max_files):
        if not os.path.exists(file[:-4] + ".webm"):
            # Use blender to render an animation of this grain rotating
            os.system(f"blender --background -t 4 --python {blender_script_path} -- " + file + " > /dev/null 2>&1")
            # Use ffmpeg to convert the animation into a webm video
            os.system(
                "ffmpeg -y -framerate 30 -pattern_type glob -i '"
                + file[:-4]
                + "/*.png' "
                + "-c:v libvpx-vp9 "
                # + '-vf "drawtext=text='+ids[i]+f':x=(w-tw)/2:y=h-th-10:fontcolor=white:fontsize=72:fontfile={font}" '
                + file[:-4]
                + ".webm"
                + silence
            )
            # Clean up the rendered images
            os.system("rm -rf " + file[:-4] + "/*.png")
            os.system("rmdir " + file[:-4])

    # Step 3: Stitch videos together into a grid
    print("Stitching videos together...")
    files = glob.glob(f"{stl_foldername}/*.webm")
    files.sort()
    num_particles = min(72, len(files))
    num_grids = int(numpy.ceil(num_particles / 12))

    for i in range(num_particles, num_grids * 12):  # pad with blank videos
        files.append(f"{stl_foldername}/blank.webm")

    print(f"    Making {num_grids} grids")
    for i in range(num_grids):
        # Make a 4x3 grid
        if not os.path.exists(f"grid_{i}.webm"):
            os.system(
                f"ffmpeg -y -i {files[i*12+0]} -i {files[i*12+1]} -i {files[i*12+ 2]} -i {files[i*12+ 3]} "
                + f"-i {files[i*12+4]} -i {files[i*12+5]} -i {files[i*12+ 6]} -i {files[i*12+ 7]} "
                + f"-i {files[i*12+8]} -i {files[i*12+9]} -i {files[i*12+10]} -i {files[i*12+11]} "
                + f'-filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v][6:v][7:v][8:v][9:v][10:v][11:v]xstack=inputs=12:layout=0_0|w0_0|w0+w1_0|w0+w1+w2_0|0_h0|w4_h0|w4+w5_h0|w4+w5+w6_h0|0_h0+h4|w8_h0+h4|w8+w9_h0+h4|w8+w9+w10_h0+h4" '
                + f"grid_{i}.webm"
                + silence
            )

    # Now concatenate the grids together temporally
    print("    Concatenating grids together...")
    with open("sources.txt", "w") as f:
        for i in range(num_grids):
            f.write(f"file grid_{i}.webm\n")
    os.system("ffmpeg -y -f concat -safe 0 -i sources.txt -c copy all_particles.webm" + silence)

    if not os.path.exists(output_foldername):
        os.makedirs(output_foldername)

    print("    Reducing file size...")
    os.system("ffmpeg -y -i all_particles.webm -crf 45 " + f"{output_foldername}/all_particles.webm" + silence)

    if not debug:
        print("Cleaning up...")
        os.system("rm grid_*.webm")


def make_individual_videos(stl_foldername, output_foldername, bg_colour=None, fg_colour=None):
    """
    Renders individual videos for each STL file in the specified folder.

    This function processes each STL file in the given folder, renders a rotating animation
    using Blender, converts the animation into a video using ffmpeg, and saves the video
    to the specified output folder.

    Args:
        stl_foldername (str): The path to the folder containing STL files.
        output_foldername (str): The path to the folder where the output videos will be saved.

    Returns:
        None

    Notes:
        - The function assumes Blender and ffmpeg are installed and available in the system's PATH.
        - The Blender script path should be defined in the variable `blender_script_path`.
        - The function uses 120 frames to create a 4-second video at 30 frames per second.
        - If the `debug` variable is set to False, the rendered images are cleaned up after the video is created.
        - The videos are named sequentially as `particle_XXXXX.mp4` in the output folder.
    """

    print("Rendering videos for IG...")
    files = glob.glob(f"{stl_foldername}/*.stl")
    files.sort()
    # print("FOUND", len(files), "FILES")
    # print(files)

    if not os.path.exists(output_foldername):
        os.makedirs(output_foldername)

    for i, file in tqdm(enumerate(files), total=len(files)):
        if not os.path.exists(file[:-4] + ".mp4"):
            # Use blender to render an animation of this grain rotating
            os.system(
                f"blender --background -t 4 --python {blender_script_path} -- "
                + f" --filename {file} " 
                + f"--frame_end 120 "  # 120 frames to make a 4 second video
                + f"--bg_colour {bg_colour} --fg_colour {fg_colour}"
                # + " > /dev/null 2>&1"
            )
            # Use ffmpeg to convert the animation into a webm video
            os.system(
                "ffmpeg -y -framerate 30 -pattern_type glob -i '"
                + file[:-4]
                + "/*.png' "
                + "-c:v libvpx-vp9 "
                # + '-vf "drawtext=text='+ids[i]+f':x=(w-tw)/2:y=h-th-10:fontcolor=white:fontsize=72:fontfile={font}" '
                + file[:-4]
                + ".mp4"
                + silence
            )
            if not debug:
                # Clean up the rendered images
                os.system("rm -rf " + file[:-4] + "/*.png")
                os.system("mv " + file[:-4] + ".mp4 " + output_foldername + "/particle_" + str(i).zfill(5) + ".mp4")
                os.system("rmdir " + file[:-4])
