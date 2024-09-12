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


def make_website_video(filename, sand_type):

    # Pass in a path to a labelled image stack
    foldername = "".join(filename.split(".")[:-1])

    if not os.path.exists("blank.webm"):
        matplotlib.image.imsave("blank.png", numpy.zeros((1000, 1000, 4)))
        os.system("ffmpeg -y -loop 1 -i blank.png -c:v libvpx-vp9 -t 2 blank.webm" + silence)

    # Step 1: convert labelled image to a set of meshes
    print("Converting labelled image to meshes...")
    # Check if there is at least one .npy file in the folder
    if glob.glob(foldername + "/npy/*.npy"):
        print("    Meshes already exist, skipping step 1")
    else:
        os.system("python labelled_image_to_mesh.py " + filename)

    # sys.exit()

    # Step 2: Render each grain using blender, then convert to a video with ffmpeg
    print("Rendering videos...")
    files = glob.glob(foldername + "/stl_ORIGINAL/*.stl")
    files.sort()

    # with open(f"../_data/sands/{sand_type}-sand.json") as f:
    #     json_data = json.load(f)
    # ids = json_data["id"].replace("'", "").split(", ")

    # ==== Changed here to render less particles
    start_index = 0  # Specify the index to start rendering from
    max_files = min(72, len(files) - start_index)  # Calculate max_files based on start_index
    for i, file in tqdm(
        enumerate(files[start_index : start_index + max_files]), total=max_files
    ):  # Adjusted range to max_files
        if not os.path.exists(file[:-4] + ".webm"):
            # Use blender to render an animation of this grain rotating
            os.system("blender --background -t 4 --python mesh_to_blender.py -- " + file + " > /dev/null 2>&1")
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

    # sys.exit()

    # Step 3: Stitch videos together into a grid
    print("Stitching videos together...")
    files = glob.glob(foldername + "/stl_ORIGINAL/*.webm")
    files.sort()
    num_particles = min(72, len(files))  # ==== Changed here to limit to 72 particles
    num_grids = int(numpy.ceil(num_particles / 12))  # Adjusted for a maximum of 72 particles

    for i in range(num_particles, num_grids * 12):  # pad with blank videos
        files.append("blank.webm")

    print(f"    Making {num_grids} grids")
    for i in range(num_grids):
        # Make a 4x3 grid
        if not os.path.exists(f"grid_{i}.webm"):
            # print(f'ffmpeg -y -i {files[i*12+0]} -i {files[i*12+1]} -i {files[i*12+ 2]} -i {files[i*12+ 3]} ' +
            #         f'-i {files[i*12+4]} -i {files[i*12+5]} -i {files[i*12+ 6]} -i {files[i*12+ 7]} ' +
            #         f'-i {files[i*12+8]} -i {files[i*12+9]} -i {files[i*12+10]} -i {files[i*12+11]} ' +
            #         f'-filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v][6:v][7:v][8:v][9:v][10:v][11:v]xstack=inputs=12:layout=0_0|w0_0|w0+w1_0|w0+w1+w2_0|0_h0|w4_h0|w4+w5_h0|w4+w5+w6_h0|0_h0+h4|w8_h0+h4|w8+w9_h0+h4|w8+w9+w10_h0+h4" ' +
            #         f'grid_{i}.webm' + silence)
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

    if not os.path.exists(foldername + "/upload"):
        os.makedirs(foldername + "/upload")

    print("    Reducing file size...")
    os.system("ffmpeg -y -i all_particles.webm -crf 45 " + foldername + "/upload/all_particles.webm" + silence)

    # Make zips for uploading
    print("    Making zips...")
    os.system("rm -rf upload/*")
    if not os.path.exists("upload"):
        os.makedirs("upload")
    for quality in ["ORIGINAL", "100", "30", "10", "3"]:
        os.system(f"zip -j upload/meshes_{quality}.zip {foldername}/stl_{quality}/*.stl")
        os.system(f"cp {foldername}/stl_{quality}/particle_00001.stl upload/ref_particle_{quality}.stl")
    os.system("zip -j upload/level_sets.zip " + foldername + "/vdb/*.vdb")

    # Step 5: Cleanup
    print("Cleaning up...")
    if not debug:
        os.system("rm grid_*.webm")

    os.system(f"mv all_particles_smaller.webm ~/code/sand-atlas/assets/sands/{sand_type}-sand/all_particles.webm")


def make_instagram_videos(filename, sand_type):

    foldername = "".join(filename.split(".")[:-1])

    print("Rendering videos for IG...")
    files = glob.glob(foldername + "/stl_ORIGINAL/*.stl")
    files.sort()

    if not os.path.exists("media"):
        os.makedirs("media")

    for i, file in tqdm(enumerate(files), total=len(files)):
        if not os.path.exists(file[:-4] + ".mp4"):
            # Use blender to render an animation of this grain rotating
            os.system(
                "blender --background -t 4 --python mesh_to_blender.py -- "
                + file
                + " 120 "  # 120 frames to make a 4 second video
                + " > /dev/null 2>&1"
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
            # Clean up the rendered images
            os.system("rm -rf " + file[:-4] + "/*.png")
            os.system("mv " + file[:-4] + ".mp4 " + "media/" + sand_type + "-" + str(i).zfill(5) + ".mp4")
            os.system("rmdir " + file[:-4])
