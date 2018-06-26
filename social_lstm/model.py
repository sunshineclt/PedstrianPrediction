import tensorflow as tf
import numpy as np
from social_lstm.grid import get_sequence_grid_mask


class SocialLSTMModel:
    def __init__(self, args, infer=False):
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        self.args = args
        self.infer = infer
        self.lstm_num = args.lstm_num
        self.grid_size = args.grid_size
        self.max_num_peds = args.max_num_peds

        # variables definition
        #############################################################################

        # LSTM cell definition
        with tf.variable_scope("LSTM_cell"):
            cell = tf.nn.rnn_cell.BasicLSTMCell(args.lstm_num, state_is_tuple=False)

        # frame * ped * (ped_id, x, y)
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[args.seq_length, args.max_num_peds, 3],
                                         name="input_data")
        # frame * ped * (ped_id, x, y)
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[args.seq_length, args.max_num_peds, 3],
                                          name="input_data")
        # frame * ped * ped * (grid * grid)
        self.grid_data = tf.placeholder(dtype=tf.float32,
                                        shape=[args.seq_length, args.max_num_peds, args.max_num_peds, args.grid_size*args.grid_size],
                                        name="grid_data")
        self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")
        self.output_size = 5

        # Define variables for the coordinate tensor embedding layer
        with tf.variable_scope("coordinate_embedding"):
            self.embedding_coord_w = tf.get_variable("embedding_coord_w", [2, args.embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.embedding_coord_b = tf.get_variable("embedding_coord_b", [args.embedding_size], initializer=tf.constant_initializer(0.1))
        # Define variables for the social tensor embedding layer
        with tf.variable_scope("tensor_embedding"):
            self.embedding_t_w = tf.get_variable("embedding_t_w", [args.grid_size*args.grid_size*args.lstm_num, args.embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.embedding_t_b = tf.get_variable("embedding_t_b", [args.embedding_size], initializer=tf.constant_initializer(0.1))
        # Define variables for the output linear layer
        with tf.variable_scope("output_layer"):
            self.output_w = tf.get_variable("output_w", [args.lstm_num, self.output_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.output_b = tf.get_variable("output_b", [self.output_size], initializer=tf.constant_initializer(0.1))
        #############################################################################

        # states definition
        #############################################################################
        with tf.variable_scope("LSTM_states"):
            self.LSTM_states = tf.zeros([args.max_num_peds, cell.state_size], name="LSTM_states")
            self.initial_states = tf.split(self.LSTM_states, args.max_num_peds, axis=0)
        with tf.variable_scope("hidden_states"):
            self.output_states = tf.split(tf.zeros(shape=[args.max_num_peds, cell.output_size]), args.max_num_peds, axis=0)
        #############################################################################

        # prepare data
        #############################################################################
        with tf.name_scope("frame_data_tensors"):
            frame_data = [tf.squeeze(input_, [0]) for input_ in tf.split(self.input_data, args.seq_length, axis=0)]
        with tf.name_scope("frame_target_data_tensors"):
            frame_target_data = [tf.squeeze(target_, [0]) for target_ in tf.split(self.target_data, args.seq_length, axis=0)]
        with tf.name_scope("grid_frame_data_tensors"):
            grid_frame_data = [tf.squeeze(input_, [0]) for input_ in tf.split(self.grid_data, args.seq_length, axis=0)]
        #############################################################################

        # other needed variables
        #############################################################################
        # Cost
        with tf.name_scope("Cost_related_stuff"):
            self.cost = tf.constant(0.0, name="cost")
            self.counter = tf.constant(0.0, name="counter")
            self.increment = tf.constant(1.0, name="increment")
        # Containers to store output distribution parameters
        with tf.name_scope("Distribution_parameters_stuff"):
            self.initial_output = tf.split(tf.zeros([args.max_num_peds, self.output_size]), args.max_num_peds, axis=0)
        # Tensor to represent non-existent ped
        with tf.name_scope("Non_existent_ped_stuff"):
            nonexistent_ped = tf.constant(0.0, name="zero_ped")
        #############################################################################

        # unfolding
        #############################################################################
        for seq, frame in enumerate(frame_data):
            current_frame_data = frame
            current_grid_frame_data = grid_frame_data[seq]

            # social_tensor = self.get_social_tensor(current_grid_frame_data)

            # spatial pyramid
            social_tensor = self.get_social_tensor_spatial_pyramid(current_grid_frame_data)

            for ped in range(args.max_num_peds):
                ped_id = current_frame_data[ped, 0]

                with tf.name_scope("extract_input_ped"):
                    self.spatial_input = tf.slice(current_frame_data, [ped, 1], [1, 2])
                    self.tensor_input = tf.slice(social_tensor, [ped, 0], [1, args.grid_size*args.grid_size*args.lstm_num])

                with tf.name_scope("embeddings_operations"):
                    # Embed the spatial input
                    embedded_spatial_input = tf.nn.relu(tf.nn.xw_plus_b(self.spatial_input, self.embedding_coord_w, self.embedding_coord_b))
                    # Embed the tensor input
                    embedded_tensor_input = tf.nn.relu(tf.nn.xw_plus_b(self.tensor_input, self.embedding_t_w, self.embedding_t_b))

                with tf.name_scope("concatenate_embeddings"):
                    # Concatenate the embeddings
                    complete_input = tf.concat([embedded_spatial_input, embedded_tensor_input], axis=1)

                with tf.variable_scope("LSTM") as scope:
                    if seq > 0 or ped > 0:
                        scope.reuse_variables()
                    self.output_states[ped], self.initial_states[ped] = cell(complete_input, self.initial_states[ped])

                with tf.name_scope("output_linear_layer"):
                    self.initial_output[ped] = tf.nn.xw_plus_b(self.output_states[ped], self.output_w, self.output_b)

                with tf.name_scope("extract_target_ped"):
                    [x_data, y_data] = tf.split(tf.slice(frame_target_data[seq], [ped, 1], [1, 2]), 2, axis=1)
                    target_ped_id = frame_target_data[seq][ped, 0]

                with tf.name_scope("get_coef"):
                    z = self.initial_output[ped]
                    o_mux, o_muy, o_sx, o_sy, o_corr = tf.split(z, 5, axis=1)
                    # The output must be exponentiated for the std devs
                    o_sx = tf.exp(o_sx)
                    o_sy = tf.exp(o_sy)
                    # Tanh applied to keep it in the range [-1, 1]
                    o_corr = tf.tanh(o_corr)

                    self.o_mux = o_mux
                    self.o_muy = o_muy
                    self.o_sx = o_sx
                    self.o_sy = o_sy
                    self.o_corr = o_corr

                with tf.name_scope("calculate_loss"):
                    # Calculate the PDF of the data w.r.t to the distribution
                    result0 = self.tf_2d_normal(x_data, y_data, o_mux, o_muy, o_sx, o_sy, o_corr)

                    # For numerical stability purposes
                    epsilon = 1e-20

                    # Apply the log operation
                    result1 = -tf.log(tf.maximum(result0, epsilon))  # Numerical stability

                    # Sum up all log probabilities for each data point
                    lossfunc = tf.reduce_sum(result1)

                with tf.name_scope("increment_cost"):
                    self.cost = tf.cond(
                        tf.logical_or(tf.equal(ped_id, nonexistent_ped), tf.equal(target_ped_id, nonexistent_ped)),
                        lambda: self.cost, lambda: tf.add(self.cost, lossfunc))
                    self.counter = tf.cond(
                        tf.logical_or(tf.equal(ped_id, nonexistent_ped), tf.equal(target_ped_id, nonexistent_ped)),
                        lambda: self.counter, lambda: tf.add(self.counter, self.increment))
        #############################################################################

        with tf.name_scope("mean_cost"):
            self.cost = tf.divide(self.cost, self.counter)

        vars = tf.trainable_variables()
        l2 = args.L2_param * sum(tf.nn.l2_loss(tvar) for tvar in vars)
        self.cost = self.cost + l2

        self.final_states = tf.concat(self.initial_states, axis = 0)

        self.final_output = self.initial_output

        self.gradients = tf.gradients(self.cost, vars)
        grads, _ = tf.clip_by_global_norm(self.gradients, args.gradient_clip)
        optimizer = tf.train.RMSPropOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, vars))

    def tf_2d_normal(self, x, y, mux, muy, sx, sy, rho):
        '''
        Function that implements the PDF of a 2D normal distribution
        params:
        x : input x points
        y : input y points
        mux : mean of the distribution in x
        muy : mean of the distribution in y
        sx : std dev of the distribution in x
        sy : std dev of the distribution in y
        rho : Correlation factor of the distribution
        '''
        # eq 3 in the paper
        # and eq 24 & 25 in Graves (2013)
        # Calculate (x - mux) and (y-muy)
        normx = tf.subtract(x, mux)
        normy = tf.subtract(y, muy)
        # Calculate sx*sy
        sxsy = tf.multiply(sx, sy)
        # Calculate the exponential factor
        z = tf.square(tf.div(normx, sx)) + tf.square(tf.div(normy, sy)) - 2*tf.div(tf.multiply(rho, tf.multiply(normx, normy)), sxsy)
        negRho = 1 - tf.square(rho)
        # Numerator
        result = tf.exp(tf.div(-z, 2*negRho))
        # Normalization constant
        denom = 2 * np.pi * tf.multiply(sxsy, tf.sqrt(negRho))
        # Final PDF calculation
        result = tf.div(result, denom)
        return result

    def get_social_tensor(self, grid_frame_data, grid_size = self.grid_size):
        '''
        Computes the social tensor for all the max_num_peds in the frame
        params:
        grid_frame_data : A tensor of shape MNP x MNP x (GS**2)
        output_states : A list of tensors each of shape 1 x RNN_size of length MNP
        '''
        # Create a zero tensor of shape MNP x (GS**2) x RNN_size
        social_tensor = tf.zeros([self.args.max_num_peds, grid_size*grid_size, self.lstm_num], name="social_tensor")
        # Create a list of zero tensors each of shape 1 x (GS**2) x RNN_size of length MNP
        social_tensor = tf.split(social_tensor, self.args.max_num_peds, axis=0)
        # Concatenate list of hidden states to form a tensor of shape MNP x RNN_size
        hidden_states = tf.concat(self.output_states, axis=0)
        # Split the grid_frame_data into grid_data for each pedestrians
        # Consists of a list of tensors each of shape 1 x MNP x (GS**2) of length MNP
        grid_frame_ped_data = tf.split(grid_frame_data, self.args.max_num_peds, axis=0)
        # Squeeze tensors to form MNP x (GS**2) matrices
        grid_frame_ped_data = [tf.squeeze(input_, [0]) for input_ in grid_frame_ped_data]

        # For each pedestrian
        for ped in range(self.args.max_num_peds):
            # Compute social tensor for the current pedestrian
            with tf.name_scope("tensor_calculation"):
                social_tensor_ped = tf.matmul(tf.transpose(grid_frame_ped_data[ped]), hidden_states)
                social_tensor[ped] = tf.reshape(social_tensor_ped, [1, grid_size*grid_size, self.lstm_num])

        # Concatenate the social tensor from a list to a tensor of shape MNP x (GS**2) x RNN_size
        social_tensor = tf.concat(social_tensor, axis=0)
        # Reshape the tensor to match the dimensions MNP x (GS**2 * RNN_size)
        social_tensor = tf.reshape(social_tensor, [self.args.max_num_peds, self.grid_size*self.grid_size*self.lstm_num])
        return social_tensor


    # the new function for getting spatial pyramid
    def get_social_tensor_spatial_pyramid(self, grid_frame_data):

        social_spatial = get_social_tensor(grid_frame_data, self.grid_size)

        for i in {2, 4}:
            social_spatial += get_social_tensor(grid_frame_data, i * self.grid_size)

        return social_spatial


    def sample_gaussian_2d(self, mux, muy, sx, sy, rho):
        '''
        Function to sample a point from a given 2D normal distribution
        params:
        mux : mean of the distribution in x
        muy : mean of the distribution in y
        sx : std dev of the distribution in x
        sy : std dev of the distribution in y
        rho : Correlation factor of the distribution
        '''
        # Extract mean
        mean = [mux, muy]
        # Extract covariance matrix
        cov = [[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]
        # Sample a point from the multivariate normal distribution
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]

    def sample(self, sess, traj, grid, dimensions, true_traj, num=10):
        # traj is a sequence of frames (of length obs_length)
        # so traj shape is (obs_length x max_num_peds x 3)
        # grid is a tensor of shape obs_length x max_num_peds x max_num_peds x (gs**2)
        states = sess.run(self.LSTM_states)
        # print "Fitting"
        # For each frame in the sequence
        for index, frame in enumerate(traj[:-1]):
            data = np.reshape(frame, (1, self.max_num_peds, 3))
            target_data = np.reshape(traj[index+1], (1, self.max_num_peds, 3))
            grid_data = np.reshape(grid[index, :], (1, self.max_num_peds, self.max_num_peds, self.grid_size*self.grid_size))

            feed = {self.input_data: data, self.LSTM_states: states, self.grid_data: grid_data, self.target_data: target_data}

            [states, cost] = sess.run([self.final_states, self.cost], feed)
            # print cost

        ret = traj

        last_frame = traj[-1]

        prev_data = np.reshape(last_frame, (1, self.max_num_peds, 3))
        prev_grid_data = np.reshape(grid[-1], (1, self.max_num_peds, self.max_num_peds, self.grid_size*self.grid_size))

        prev_target_data = np.reshape(true_traj[traj.shape[0]], (1, self.max_num_peds, 3))
        # Prediction
        for t in range(num):
            # print "**** NEW PREDICTION TIME STEP", t, "****"
            feed = {self.input_data: prev_data, self.LSTM_states: states, self.grid_data: prev_grid_data, self.target_data: prev_target_data}
            [output, states, cost] = sess.run([self.final_output, self.final_states, self.cost], feed)
            # print "Cost", cost
            # Output is a list of lists where the inner lists contain matrices of shape 1x5. The outer list contains only one element (since seq_length=1) and the inner list contains max_num_peds elements
            # output = output[0]
            newpos = np.zeros((1, self.max_num_peds, 3))
            for pedindex, pedoutput in enumerate(output):
                [o_mux, o_muy, o_sx, o_sy, o_corr] = np.split(pedoutput[0], 5, 0)
                mux, muy, sx, sy, corr = o_mux[0], o_muy[0], np.exp(o_sx[0]), np.exp(o_sy[0]), np.tanh(o_corr[0])

                next_x, next_y = self.sample_gaussian_2d(mux, muy, sx, sy, corr)

                # if prev_data[0, pedindex, 0] != 0:
                #     print "Pedestrian ID", prev_data[0, pedindex, 0]
                #     print "Predicted parameters", mux, muy, sx, sy, corr
                #     print "New Position", next_x, next_y
                #     print "Target Position", prev_target_data[0, pedindex, 1], prev_target_data[0, pedindex, 2]
                #     print

                newpos[0, pedindex, :] = [prev_data[0, pedindex, 0], next_x, next_y]
            ret = np.vstack((ret, newpos))
            prev_data = newpos
            prev_grid_data = get_sequence_grid_mask(prev_data, dimensions, self.args.neighborhood_size, self.grid_size)
            if t != num - 1:
                prev_target_data = np.reshape(true_traj[traj.shape[0] + t + 1], (1, self.max_num_peds, 3))

        # The returned ret is of shape (obs_length+pred_length) x max_num_peds x 3
        return ret

