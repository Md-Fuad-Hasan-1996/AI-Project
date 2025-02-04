# import matplotlib.pyplot as plt
# import os
# from typing import Tuple, Sequence, Dict, Union, Optional, Callable
# import numpy as np
# import torch
# import torch.nn as nn
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# import matplotlib.pyplot as plt
# import yaml
# from vint_train.training.train_utils import get_action

# # ROS
# import rospy
# from sensor_msgs.msg import Image
# from std_msgs.msg import Bool, Float32MultiArray
# from utils import msg_to_pil, to_numpy, transform_images, load_model


# import torch
# from PIL import Image as PILImage
# import numpy as np
# import argparse
# import yaml
# import time

# # UTILS
# from topic_names import (IMAGE_TOPIC,
#                         WAYPOINT_TOPIC,
#                         SAMPLED_ACTIONS_TOPIC)


# # CONSTANTS
# TOPOMAP_IMAGES_DIR = "../topomaps/images"
# MODEL_WEIGHTS_PATH = "../model_weights"
# ROBOT_CONFIG_PATH ="../config/robot.yaml"
# MODEL_CONFIG_PATH = "../config/models.yaml"
# with open(ROBOT_CONFIG_PATH, "r") as f:
#     robot_config = yaml.safe_load(f)
# MAX_V = robot_config["max_v"]
# MAX_W = robot_config["max_w"]
# RATE = robot_config["frame_rate"] 

# # GLOBALS
# context_queue = []
# context_size = None  
# subgoal = []

# # Load the model 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)


# def callback_obs(msg):
#     obs_img = msg_to_pil(msg)
#     if context_size is not None:
#         if len(context_queue) < context_size + 1:
#             context_queue.append(obs_img)
#         else:
#             context_queue.pop(0)
#             context_queue.append(obs_img)


# def main(args: argparse.Namespace):
#     global context_size

#      # load model parameters
#     with open(MODEL_CONFIG_PATH, "r") as f:
#         model_paths = yaml.safe_load(f)

#     model_config_path = model_paths[args.model]["config_path"]
#     with open(model_config_path, "r") as f:
#         model_params = yaml.safe_load(f)

#     context_size = model_params["context_size"]

#     # load model weights
#     ckpth_path = model_paths[args.model]["ckpt_path"]
#     if os.path.exists(ckpth_path):
#         print(f"Loading model from {ckpth_path}")
#     else:
#         raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
#     model = load_model(
#         ckpth_path,
#         model_params,
#         device,
#     )
#     model = model.to(device)
#     model.eval()

    
#      # load topomap
#     topomap_filenames = sorted(os.listdir(os.path.join(
#         TOPOMAP_IMAGES_DIR, args.dir)), key=lambda x: int(x.split(".")[0]))
#     topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
#     num_nodes = len(os.listdir(topomap_dir))
#     topomap = []
#     for i in range(num_nodes):
#         image_path = os.path.join(topomap_dir, topomap_filenames[i])
#         topomap.append(PILImage.open(image_path))

#     closest_node = 0
#     assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
#     if args.goal_node == -1:
#         goal_node = len(topomap) - 1
#     else:
#         goal_node = args.goal_node
#     reached_goal = False

#      # ROS
#     rospy.init_node("EXPLORATION", anonymous=False)
#     rate = rospy.Rate(RATE)
#     image_curr_msg = rospy.Subscriber(
#         IMAGE_TOPIC, Image, callback_obs, queue_size=1)
#     waypoint_pub = rospy.Publisher(
#         WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)  
#     sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
#     goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)

#     print("Registered with master node. Waiting for image observations...")

#     if model_params["model_type"] == "nomad":
#         num_diffusion_iters = model_params["num_diffusion_iters"]
#         noise_scheduler = DDPMScheduler(
#             num_train_timesteps=model_params["num_diffusion_iters"],
#             beta_schedule='squaredcos_cap_v2',
#             clip_sample=True,
#             prediction_type='epsilon'
#         )
#     # navigation loop
#     while not rospy.is_shutdown():
#         # EXPLORATION MODE
#         chosen_waypoint = np.zeros(4)
#         if len(context_queue) > model_params["context_size"]:
#             if model_params["model_type"] == "nomad":
#                 obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
#                 obs_images = torch.split(obs_images, 3, dim=1)
#                 obs_images = torch.cat(obs_images, dim=1) 
#                 obs_images = obs_images.to(device)
#                 mask = torch.zeros(1).long().to(device)  

#                 start = max(closest_node - args.radius, 0)
#                 end = min(closest_node + args.radius + 1, goal_node)
#                 goal_image = [transform_images(g_img, model_params["image_size"], center_crop=False).to(device) for g_img in topomap[start:end + 1]]
#                 goal_image = torch.concat(goal_image, dim=0)

#                 obsgoal_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
#                 dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
#                 dists = to_numpy(dists.flatten())
#                 min_idx = np.argmin(dists)
#                 closest_node = min_idx + start
#                 print("closest node:", closest_node)
#                 sg_idx = min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)
#                 obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

#                 # infer action
#                 with torch.no_grad():
#                     # encoder vision features
#                     if len(obs_cond.shape) == 2:
#                         obs_cond = obs_cond.repeat(args.num_samples, 1)
#                     else:
#                         obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
                    
#                     # initialize action from Gaussian noise
#                     noisy_action = torch.randn(
#                         (args.num_samples, model_params["len_traj_pred"], 2), device=device)
#                     naction = noisy_action

#                     # init scheduler
#                     noise_scheduler.set_timesteps(num_diffusion_iters)

#                     start_time = time.time()
#                     for k in noise_scheduler.timesteps[:]:
#                         # predict noise
#                         noise_pred = model(
#                             'noise_pred_net',
#                             sample=naction,
#                             timestep=k,
#                             global_cond=obs_cond
#                         )
#                         # inverse diffusion step (remove noise)
#                         naction = noise_scheduler.step(
#                             model_output=noise_pred,
#                             timestep=k,
#                             sample=naction
#                         ).prev_sample
#                     print("time elapsed:", time.time() - start_time)

#                 naction = to_numpy(get_action(naction))
#                 sampled_actions_msg = Float32MultiArray()
#                 sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten()))
#                 print("published sampled actions")
#                 sampled_actions_pub.publish(sampled_actions_msg)
#                 naction = naction[0] 
#                 chosen_waypoint = naction[args.waypoint]
#             else:
#                 start = max(closest_node - args.radius, 0)
#                 end = min(closest_node + args.radius + 1, goal_node)
#                 distances = []
#                 waypoints = []
#                 batch_obs_imgs = []
#                 batch_goal_data = []
#                 for i, sg_img in enumerate(topomap[start: end + 1]):
#                     transf_obs_img = transform_images(context_queue, model_params["image_size"])
#                     goal_data = transform_images(sg_img, model_params["image_size"])
#                     batch_obs_imgs.append(transf_obs_img)
#                     batch_goal_data.append(goal_data)
                    
#                 # predict distances and waypoints
#                 batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
#                 batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)

#                 distances, waypoints = model(batch_obs_imgs, batch_goal_data)
#                 distances = to_numpy(distances)
#                 waypoints = to_numpy(waypoints)
#                 # look for closest node
#                 min_dist_idx = np.argmin(distances)
#                 # chose subgoal and output waypoints
#                 if distances[min_dist_idx] > args.close_threshold:
#                     chosen_waypoint = waypoints[min_dist_idx][args.waypoint]
#                     closest_node = start + min_dist_idx
#                 else:
#                     chosen_waypoint = waypoints[min(
#                         min_dist_idx + 1, len(waypoints) - 1)][args.waypoint]
#                     closest_node = min(start + min_dist_idx + 1, goal_node)
#         # RECOVERY MODE
#         if model_params["normalize"]:
#             chosen_waypoint[:2] *= (MAX_V / RATE)  
#         waypoint_msg = Float32MultiArray()
#         waypoint_msg.data = chosen_waypoint
#         waypoint_pub.publish(waypoint_msg)
#         reached_goal = closest_node == goal_node
#         goal_pub.publish(reached_goal)
#         if reached_goal:
#             print("Reached goal! Stopping...")
#         rate.sleep()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
#     parser.add_argument(
#         "--model",
#         "-m",
#         default="nomad",
#         type=str,
#         help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
#     )
#     parser.add_argument(
#         "--waypoint",
#         "-w",
#         default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
#         type=int,
#         help=f"""index of the waypoint used for navigation (between 0 and 4 or 
#         how many waypoints your model predicts) (default: 2)""",
#     )
#     parser.add_argument(
#         "--dir",
#         "-d",
#         default="topomap",
#         type=str,
#         help="path to topomap images",
#     )
#     parser.add_argument(
#         "--goal-node",
#         "-g",
#         default=-1,
#         type=int,
#         help="""goal node index in the topomap (if -1, then the goal node is 
#         the last node in the topomap) (default: -1)""",
#     )
#     parser.add_argument(
#         "--close-threshold",
#         "-t",
#         default=3,
#         type=int,
#         help="""temporal distance within the next node in the topomap before 
#         localizing to it (default: 3)""",
#     )
#     parser.add_argument(
#         "--radius",
#         "-r",
#         default=4,
#         type=int,
#         help="""temporal number of locobal nodes to look at in the topopmap for
#         localization (default: 2)""",
#     )
#     parser.add_argument(
#         "--num-samples",
#         "-n",
#         default=8,
#         type=int,
#         help=f"Number of actions sampled from the exploration model (default: 8)",
#     )
#     args = parser.parse_args()
#     print(f"Using {device}")
#     main(args)


import os
import yaml
import pickle
import torch
import numpy as np
from PIL import Image as PILImage
from vint_train.training.train_utils import get_action
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from utils import to_numpy, transform_images, load_model
import logging

# Constants
DATASET_DIR = "../dataset"
MODEL_CONFIG_PATH = "../config/models.yaml"
LOG_FILE = "offline_navigation_log.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def setup_logger(log_file):
    """Set up the logger to write to a file."""
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Logger initialized.")

def load_trajectory_data(trajectory_dir):
    """Load images and trajectory metadata from a trajectory directory."""
    images = sorted(
        [os.path.join(trajectory_dir, f) for f in os.listdir(trajectory_dir) if f.endswith(".jpg")],
        key=lambda x: int(os.path.basename(x).split(".")[0]),
    )
    traj_data_path = os.path.join(trajectory_dir, "traj_data.pkl")
    with open(traj_data_path, "rb") as f:
        traj_data = pickle.load(f)
    return images, traj_data

def main():
    # Initialize logger
    setup_logger(LOG_FILE)

    # Load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)
    model_name = "nomad"  # Pre-trained model name
    model_config_path = model_paths[model_name]["config_path"]

    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    # Load model weights
    ckpth_path = model_paths[model_name]["ckpt_path"]
    if not os.path.exists(ckpth_path):
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    logging.info(f"Loading model from {ckpth_path}")
    model = load_model(ckpth_path, model_params, device).to(device).eval()

    # Initialize scheduler for diffusion-based models
    if model_params["model_type"] == "nomad":
        num_diffusion_iters = model_params["num_diffusion_iters"]
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

    # Iterate through dataset
    for trajectory_name in os.listdir(DATASET_DIR):
        trajectory_dir = os.path.join(DATASET_DIR, trajectory_name)
        if not os.path.isdir(trajectory_dir):
            continue

        logging.info(f"Processing trajectory: {trajectory_name}")

        # Load images and trajectory data
        images, traj_data = load_trajectory_data(trajectory_dir)
        context_queue = []
        context_size = model_params["context_size"]

        for step, img_path in enumerate(images):
            img = PILImage.open(img_path)
            context_queue.append(transform_images(img, model_params["image_size"], center_crop=False))

            if len(context_queue) > context_size + 1:
                context_queue.pop(0)

            if len(context_queue) >= context_size:
                obs_images = torch.cat(context_queue, dim=1).to(device)

                # Prepare goal image (e.g., the last image in the trajectory)
                goal_image = transform_images(
                    PILImage.open(images[-1]), model_params["image_size"], center_crop=False
                ).to(device)

                # Create input for vision_encoder (6-channel tensor)
                last_obs_img = obs_images[:, -3:, :, :]  # Last 3 channels
                obsgoal_img = torch.cat((last_obs_img, goal_image), dim=1)  # Combine to make 6 channels

                # Process observations through the model
                obsgoal_cond = model(
                    "vision_encoder",
                    obs_img=obsgoal_img,
                    goal_img=goal_image,
                    input_goal_mask=torch.zeros(len(goal_image), dtype=torch.long).to(device)
                )

                dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                dists = to_numpy(dists.flatten())

                # Log distances
                logging.info(f"Trajectory {trajectory_name}, Step {step}: Distances - {dists}")

                # Sample actions with diffusion
                noisy_action = torch.randn((8, model_params["len_traj_pred"], 2), device=device)
                for k in noise_scheduler.timesteps:
                    noise_pred = model(
                        "noise_pred_net", sample=noisy_action, timestep=k, global_cond=obsgoal_cond
                    )
                    noisy_action = noise_scheduler.step(
                        model_output=noise_pred, timestep=k, sample=noisy_action
                    ).prev_sample

                # Decode and log actions
                naction = to_numpy(get_action(noisy_action))[0]
                logging.info(f"Trajectory {trajectory_name}, Step {step}: Sampled action - {naction}")

if __name__ == "__main__":
    main()


