import os
import numpy as np
import pickle
import random


class DataLoader:

    def __init__(self,
                 batch_size,
                 seq_length,
                 max_num_peds,
                 force_pre_process=False,
                 infer=False):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.max_num_peds = max_num_peds
        self.infer = infer

        self.validate_fraction = 0.2

        original_data_path = "../data/pixel_pos.csv"
        transformed_data_path = "../data/transformed_data.pkl"

        # all_frame_data contains data (except validation data) with shape [frame, ped, 3] and 3 is ID, x, y
        self.training_frame_data = None
        # what frames do data have
        self.frame_list = None
        # how many peds are in each framex
        self.num_peds_data = None
        # validate_frame_data has shape [frame, ped, 3] and 3 is ID, x, y
        self.validate_frame_data = None
        self.num_training_batch = 0
        self.num_validate_batch = 0

        if not os.path.exists(transformed_data_path) or force_pre_process:
            self.preprocess(original_data_path, transformed_data_path)

        self.load_preprocess(transformed_data_path)

        self.training_frame_pointer = 0
        self.validate_frame_pointer = 0
        self.reset_batch_pointer(validate=False)
        self.reset_batch_pointer(validate=True)

    def preprocess(self, original_data_path, transformed_data_path):
        data = np.genfromtxt(original_data_path, delimiter=",")
        # what frames do data have
        frame_list = np.unique(data[0, :]).tolist()
        num_frames = len(frame_list)
        if self.infer:
            validate_num_frames = 0
        else:
            validate_num_frames = int(num_frames * self.validate_fraction)
        # training_frame_data contains data (except validation data) with shape [frame, ped, 3] and 3 is ID, x, y
        training_frame_data = np.zeros(shape=[num_frames - validate_num_frames, self.max_num_peds, 3])
        # validate_frame_data has shape [frame, ped, 3] and 3 is ID, x, y
        validate_frame_data = np.zeros(shape=[validate_num_frames, self.max_num_peds, 3])
        # how many peds are in each framex
        num_peds_data = []
        for index, frame in enumerate(frame_list):
            this_frame_data = data[:, data[0, :] == frame]
            peds_in_this_frame = this_frame_data[1, :].tolist()
            num_peds_data.append(len(peds_in_this_frame))

            ped_pos = []
            for ped in peds_in_this_frame:
                current_x = this_frame_data[3, this_frame_data[1, :] == ped][0]
                current_y = this_frame_data[2, this_frame_data[1, :] == ped][0]
                ped_pos.append((ped, current_x, current_y))

            ped_pos = np.array(ped_pos)
            if (index < num_frames - validate_num_frames) or self.infer:
                training_frame_data[index, :len(peds_in_this_frame), :] = ped_pos
            else:
                validate_frame_data[index - (num_frames - validate_num_frames), :len(peds_in_this_frame), :] = ped_pos

        f = open(transformed_data_path, "wb")
        pickle.dump((training_frame_data, frame_list, num_peds_data, validate_frame_data), f)
        f.close()

    def load_preprocess(self, transformed_data_path):
        f = open(transformed_data_path, "rb")
        raw_data = pickle.load(f)
        f.close()

        self.training_frame_data = raw_data[0]
        self.frame_list = raw_data[1]
        self.num_peds_data = raw_data[2]
        self.validate_frame_data = raw_data[3]

        number_of_training = len(self.training_frame_data)
        number_of_validate = len(self.validate_frame_data)
        self.num_training_batch = int(number_of_training / self.batch_size) * 2 # because of the random choose
        self.num_validate_batch = int(number_of_validate / self.batch_size)

    def next_training_batch(self, random_choose=True):
        x_batch = []
        y_batch = []
        i = 0
        while i < self.batch_size:
            index = self.training_frame_pointer
            if index + self.seq_length + 1 < self.training_frame_data.shape[0]:
                seq_frame_data = self.training_frame_data[index:index+self.seq_length+1, :]
                seq_source_frame_data = self.training_frame_data[index:index+self.seq_length, :]
                seq_target_frame_data = self.training_frame_data[index+1:index+self.seq_length+1, :]
                ped_in_sequence = np.unique(seq_frame_data[:, :, 0])

                source_data = np.zeros((self.seq_length, self.max_num_peds, 3))
                target_data = np.zeros((self.seq_length, self.max_num_peds, 3))

                for seq in range(self.seq_length):
                    this_seq_source_frame_data = seq_source_frame_data[seq, :]
                    this_seq_target_frame_data = seq_target_frame_data[seq, :]
                    for index, ped_id in enumerate(ped_in_sequence):
                        if ped_id == 0:
                            continue
                        else:
                            source_temp = this_seq_source_frame_data[this_seq_source_frame_data[:, 0] == ped_id, :]
                            target_temp = this_seq_target_frame_data[this_seq_target_frame_data[:, 0] == ped_id, :]
                            if source_temp.size != 0:
                                source_data[seq, index, :] = source_temp
                            if target_temp.size != 0:
                                target_data[seq, index, :] = target_temp

                x_batch.append(source_data)
                y_batch.append(target_data)

                if random_choose:
                    self.training_frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.training_frame_pointer += self.seq_length

                i += 1
            else:
                self.training_frame_pointer = 0

        return x_batch, y_batch

    def next_validate_batch(self, random_choose=True):
        x_batch = []
        y_batch = []
        i = 0
        while i < self.batch_size:
            index = self.validate_frame_pointer
            if index + self.seq_length + 1 < self.validate_frame_data.shape[0]:
                seq_frame_data = self.validate_frame_data[index:index+self.seq_length+1, :]
                seq_source_frame_data = self.validate_frame_data[index:index+self.seq_length, :]
                seq_target_frame_data = self.validate_frame_data[index+1:index+self.seq_length+1, :]
                ped_in_sequence = np.unique(seq_frame_data[:, :, 0])

                source_data = np.zeros((self.seq_length, self.max_num_peds, 3))
                target_data = np.zeros((self.seq_length, self.max_num_peds, 3))

                for seq in range(self.seq_length):
                    this_seq_source_frame_data = seq_source_frame_data[seq, :]
                    this_seq_target_frame_data = seq_target_frame_data[seq, :]
                    for index, ped_id in enumerate(ped_in_sequence):
                        if ped_id == 0:
                            continue
                        else:
                            source_temp = this_seq_source_frame_data[this_seq_source_frame_data[:, 0] == ped_id, :]
                            target_temp = this_seq_target_frame_data[this_seq_target_frame_data[:, 0] == ped_id, :]
                            if source_temp.size != 0:
                                source_data[seq, index, :] = source_temp
                            if target_temp.size != 0:
                                target_data[seq, index, :] = target_temp

                x_batch.append(source_data)
                y_batch.append(target_data)

                if random_choose:
                    self.validate_frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.validate_frame_pointer += self.seq_length

                i += 1
            else:
                self.validate_frame_pointer = 0

        return x_batch, y_batch

    def reset_batch_pointer(self, validate):
        if validate:
            self.validate_frame_pointer = 0
        else:
            self.training_frame_pointer = 0
