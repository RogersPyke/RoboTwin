## 本文件用于将robotwin Challenge 2 中的hdf5数据转为TinyVLA可以直接训练的数据。
import sys

sys.path.append('./policy/ACT/')

import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import pdb

task_prompt = {
    "adjust_bottle": "Adjust the bottle to the target pose using the manipulator.",
    "beat_block_hammer": "Use the hammer to strike the target block.",
    "blocks_ranking_rgb": "Sort or rank blocks according to their colors.",
    "blocks_ranking_size": "Sort or rank blocks according to their sizes.",
    "bottle_adjust": "Adjust the bottle to the target pose using the manipulator.",
    "click_alarmclock": "Interact with the alarm clock control.",
    "click_bell": "Press or ring the bell.",
    "dump_bin_bigbin": "Dump contents from the small bin into the large bin.",
    "grab_roller": "Grasp and manipulate the roller to the goal.",
    "handover_block": "Hand the block from one arm to the other and complete the placement.",
    "handover_mic": "Hand the microphone from one arm to the other and complete the task.",
    "hanging_mug": "Use dual arms: grasp the mug, hand off, and hang it onto the rack hook.",
    "hanging_mug__unhanging_mug": "Either hang the mug on the rack or take it off the rack and place it on the table, depending on the episode task.",
    "lift_pot": "Lift the pot to the target pose.",
    "move_can_pot": "Move the can and place it with respect to the pot.",
    "move_pillbottle_pad": "Pick up the pill bottle and place it on the target pad.",
    "move_playingcard_away": "Move the playing card away to the designated region.",
    "move_stapler_pad": "Move the stapler onto the target pad.",
    "open_laptop": "Open the laptop lid.",
    "open_microwave": "Open the microwave door.",
    "pick_diverse_bottles": "Pick valid bottles from a diverse set to the goal.",
    "pick_dual_bottles": "Pick two bottles and satisfy the placement goal.",
    "place_a2b_left": "Move the object from region A to region B using left-arm-centric motions.",
    "place_a2b_right": "Move the object from region A to region B using right-arm-centric motions.",
    "place_bread_basket": "Place the bread into the basket.",
    "place_bread_skillet": "Place the bread onto the skillet.",
    "place_burger_fries": "Arrange the burger and fries at the target layout.",
    "place_can_basket": "Place the can into the basket.",
    "place_cans_plasticbox": "Place cans into the plastic box.",
    "place_container_plate": "Place the container onto the plate.",
    "place_dual_shoes": "Place both shoes at their target poses.",
    "place_empty_cup": "Place the empty cup at the target location.",
    "place_fan": "Place the fan at the target pose.",
    "place_mouse_pad": "Place the mouse onto the mouse pad.",
    "place_object_basket": "Place the object into the basket.",
    "place_object_scale": "Use one arm to grab the object and put it on the scale.",
    "place_object_stand": "Place the object onto the stand.",
    "place_phone_stand": "Place the phone onto the stand using desk camera views to plan the motion.",
    "place_shoe": "Place the shoe at the target pose.",
    "press_stapler": "Press the stapler to staple.",
    "put_bottles_dustbin": "Put the bottles into the dustbin.",
    "put_object_cabinet": "Put the object inside the cabinet.",
    "rotate_qrcode": "Rotate the QR code object to the target orientation.",
    "scan_object": "Execute the scan motion or viewpoint sequence for the object.",
    "shake_bottle": "Shake the bottle as required by the task.",
    "shake_bottle_horizontally": "Shake the bottle with horizontal motion.",
    "stack_blocks_three": "Stack three blocks stably.",
    "stack_blocks_two": "Stack two blocks stably.",
    "stack_bowls_three": "Stack three bowls stably.",
    "stack_bowls_two": "Stack two bowls stably.",
    "stamp_seal": "Use the seal to stamp.",
    "turn_switch": "Turn the switch to the target state.",
    "unhanging_mug": "Take the mug off the rack and place it on the table using dual-arm coordination.",
    "unmove_pillbottle_pad": "Pick up the pill bottle from the pad and return it to the table region.",
    "unstack_blocks_three": "Unstack three blocks safely to separate them.",
    "unstack_bowls_three": "Unstack three bowls safely to separate them.",
}

def load_hdf5(dataset_path):
    '''
    从robotwin Challenge 2 生成的 hdf5文件中读取数据
    '''
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        left_gripper, left_arm = root['/joint_action/left_gripper'][()], root['/joint_action/left_arm'][()]
        right_gripper, right_arm = root['/joint_action/right_gripper'][()], root['/joint_action/right_arm'][()]
        image_dict = dict() 
        for cam_name in root[f'/observation/'].keys():
            image_dict[cam_name] = root[f'/observation/{cam_name}/rgb'][()] 

    return left_gripper, left_arm, right_gripper, right_arm, image_dict



def data_transform(path, episode_num, save_path, task_name):
    '''
    将原始数据转换为 VLA 模型可以使用的格式，并保存为新的 HDF5 文件。
    '''
    begin = 0
    floders = os.listdir(path)  
    assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = load_hdf5(
            os.path.join(path, f"episode{i}.hdf5"))
        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        last_state = None
        for j in range(0, left_gripper_all.shape[0]):

            left_gripper, left_arm, right_gripper, right_arm = left_gripper_all[j], left_arm_all[j], right_gripper_all[
                j], right_arm_all[j],

            if j != left_gripper_all.shape[0] - 1:
                state = np.concatenate((left_arm, [left_gripper], right_arm, [right_gripper]), axis=0)  # joint

                state = state.astype(np.float32)
                qpos.append(state)

                camera_high_bits = image_dict['head_camera'][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (640, 480))
                cam_high.append(camera_high_resized)

                camera_right_wrist_bits = image_dict['right_camera'][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                camera_left_wrist_bits = image_dict['left_camera'][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            if j != 0:
                action = state
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        hdf5path = os.path.join(save_path, f'episode_{i}.hdf5')

        with h5py.File(hdf5path, 'w') as f:
            f.create_dataset('action', data=np.array(actions))
            language_raw = task_prompt[task_name].encode('utf-8')
            f.create_dataset('language_raw', data=np.array(language_raw))
            obs = f.create_group('observations')
            obs.create_dataset('qpos', data=np.array(qpos))
            obs.create_dataset('qvel', data=np.array(qpos)) # 无意义为了对齐key
            obs.create_dataset('left_arm_dim', data=np.array(left_arm_dim))
            obs.create_dataset('right_arm_dim', data=np.array(right_arm_dim))
            image = obs.create_group('images')
            image.create_dataset('cam_high', data=np.stack(cam_high), dtype=np.uint8)
            image.create_dataset('cam_right_wrist', data=np.stack(cam_right_wrist), dtype=np.uint8)
            image.create_dataset('cam_left_wrist', data=np.stack(cam_left_wrist), dtype=np.uint8)

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some episodes.')
    parser.add_argument('task_name', type=str, default='bottle_adjust',
                        help='The name of the task (e.g., bottle_adjust)')
    parser.add_argument('setting', type=str)
    parser.add_argument('expert_data_num', type=int, default=50,
                        help='Number of episodes to process (e.g., 50)')

    args = parser.parse_args()

    task_name = args.task_name
    setting = args.setting
    expert_data_num = args.expert_data_num

    data_path_name = task_name + "/" + setting + "/data"
    begin = 0
    begin = data_transform(os.path.join("../../data/", data_path_name), expert_data_num,
                           f"data/sim-{task_name}/{setting}-{expert_data_num}",task_name)
