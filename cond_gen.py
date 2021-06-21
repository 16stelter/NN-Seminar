from zipfile import ZipFile
import numpy as np
import regex as re
import os
import json
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, SimpleRNN, GRU, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TimeDistributed, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import tensorflow.keras.backend as K
from datetime import datetime


class ConditionalGenerator():

    def __init__(self):
        # data params
        self.samples = {"data": []}
        self.filenames = []
        self.path = './csmdata/data'
        self.unused_samples = []

        # general training params
        self.epochs = 10000
        self.batch_size = 10
        self.lr = 0.001
        self.decay = 0.0
        self.loss_weights = [1.0, 0.001]
        self.alpha = K.variable(1.0, name='alpha')

        # sample shapes
        self.cut_length = 200
        self.step_size = 50
        self.dof = 99
        self.nb_label = 10
        self.latent_dim = 50

        # model params
        self.dropout_dis_list = [0.0, 0.0]
        self.hidden_dim_enc_list = [100, 100]
        self.activation_enc_list = ['tanh', 'tanh']
        self.hidden_dim_dec_list = [100, 100]
        self.activation_dec_list = ['tanh', 'tanh']
        self.hidden_dim_dis_list = [100, 40]
        self.activation_dis_list = ['relu', 'relu']
        self.latent_activation = 'tanh'

        # models
        self.decoder = self.Decoder()
        self.encoder = self.Encoder()
        self.discmt = self.Discriminator()
        self.casae = self.CASAE()

        self.compile()

    def Encoder(self):
        """
        Defines the Encoder model
        :return: keras model
        """
        motion_input = Input(shape=(self.cut_length, self.dof), name='encoder_input')
        label_input = Input(batch_size=self.batch_size, shape=(self.nb_label,), name='label_input_enc')
        label_seq = label_input
        label_seq = RepeatVector(self.cut_length)(label_seq)
        encoded = Concatenate(axis=2)([motion_input, label_seq])
        for i, (dim, activation) in enumerate(zip(self.hidden_dim_enc_list, self.activation_enc_list)):
            encoded = LSTM(units=dim, activation=activation, return_sequences=True)(encoded)

        encoded = LSTM(units=self.latent_dim, activation=self.latent_activation, name='encoded_layer',
                       return_sequences=False)(encoded)
        encoded = Dense(units=self.latent_dim, activation='linear')(encoded)

        return Model(inputs=[motion_input, label_input], outputs=encoded, name='Encoder')

    def Decoder(self):
        """
        Defines the decoder model
        :return: keras model
        """
        latent_input = Input(batch_size=self.batch_size, shape=(self.latent_dim,), name='latent_input')
        latent_input_seq = RepeatVector(self.cut_length)(latent_input)

        label_input = Input(batch_size=self.batch_size, shape=(self.nb_label,), name='label_input_dec')

        label_seq = label_input
        label_seq = RepeatVector(self.cut_length)(label_seq)
        decoded = Concatenate(axis=2)([latent_input_seq, label_seq])

        for i, (dim, activation) in enumerate(zip(self.hidden_dim_dec_list, self.activation_dec_list)):
            decoded = LSTM(units=dim, activation=activation, return_sequences=True)(decoded)
        decoded = SimpleRNN(units=self.dof, activation='sigmoid', name='decoder_output', return_sequences=True)(decoded)
        return Model(inputs=[latent_input, label_input], outputs=decoded, name='Decoder')

    def Discriminator(self):
        """
        Defines the discriminator model
        :return: keras model
        """
        input = Input(shape=(self.latent_dim,), name='discmt_input')
        for i, (dim, activation, dropout) in enumerate(
                zip(self.hidden_dim_dis_list, self.activation_dis_list, self.dropout_dis_list)):
            if i == 0:
                discmt = Dense(dim, activation=activation)(input)
            else:
                discmt = Dropout(dropout)(discmt)
                discmt = Dense(dim, activation=activation)(discmt)

        discmt = Dense(1, activation='sigmoid', name='discmt_output')(discmt)
        return Model(inputs=input, outputs=discmt, name='Discmt')

    def CASAE(self):
        """
        Combines all models to a conditional adversarial sequence autoencoder
        :return: keras model
        """
        casae = self.decoder([self.encoder.output, self.decoder.inputs[1]])
        self.discmt.trainable = False
        aux_output_discmt = self.discmt(self.encoder.output)
        return Model(inputs=[self.encoder.inputs, self.decoder.inputs[1]], outputs=[casae, aux_output_discmt],
                     name="CASAE")

    def compile(self):
        """
        Compiles the network
        """
        optimizer_discmt = SGD(lr=self.lr, decay=self.decay)
        optimizer_casae = SGD(lr=self.lr, decay=self.decay)
        self.discmt.trainable = True
        self.discmt.compile(optimizer_discmt, loss='binary_crossentropy', metrics=['accuracy'])
        self.casae.compile(optimizer_casae, loss={'Decoder': self.loss_mse_velocity_loss,
                                                  'Discmt': 'binary_crossentropy'},
                           loss_weights=self.loss_weights,
                           metrics={'Decoder': 'mse'})
        print(self.decoder.summary())
        print(self.casae.summary())

    def loss_mse_velocity_loss(self, y_true, y_pred):
        """
        Helper method for optimizers.
        """
        mse = K.mean(K.square(y_pred - y_true))
        mse_v = K.mean(K.square((y_pred[:, 1:, :] - y_pred[:, 0:-1, :]) - (y_true[:, 1:, :] - y_true[:, 0:-1, :])))
        return mse + self.alpha * mse_v

    def store_filenames(self):
        """
        Reads all filenames in the dataset, as we can't save the contents of the files in RAM.
        """
        print("Reading data...")
        for filename in os.listdir(self.path):
            if filename.endswith('.csm'):
                self.filenames.append(filename)

    def read_data(self, filename):
        """
        Converts data in a single file into a format that can be used for training. Splits whole motion into chunks of
        length self.cut_length
        :param filename: The name of the file to read
        :return: Training samples from that file
        """
        # reads data of single file, as we can't store everything at the same time
        self.samples["data"] = []
        with open(os.path.join(self.path, filename), 'r') as data:
            filename = filename.split('_')
            if len(filename) > 5:  # catchall for weird filenames
                del filename[0]
            # get emotion from filename as one hot vector
            emotions = {
                "af": [1, 0, 0, 0, 0],
                "an": [0, 1, 0, 0, 0],
                "ha": [0, 0, 1, 0, 0],
                "nu": [0, 0, 0, 1, 0],
                "sa": [0, 0, 0, 0, 1]
            }
            emotion = emotions[filename[2]]
            # get motion type from filename as one hot vector
            motions = {
                "knock": [1, 0, 0, 0, 0],
                "lift": [0, 1, 0, 0, 0],
                "seq": [0, 0, 1, 0, 0],
                "throw": [0, 0, 0, 1, 0],
                "walk": [0, 0, 0, 0, 1]
            }
            motion = motions[filename[1]]
            content = data.read()
            # select every word that starts with $
            sections = re.split('\$\w+', content)
            # split order entry on spaces, remove empty elements and cast back to list
            # this gives the order of the joints in the samples data
            order = list(filter(None, re.split('\s', sections[11])))
            sequence = []
            for frame in list(filter(None, re.split('\n', sections[12]))):  # split on newline, remove empty
                # split on blank space, remove empty and 'DROPOUT', convert to list, then map to float and back to list...
                parts = list(map(float, list(filter(None, [x.replace('DROPOUT', '') for x in re.split('\s', frame)]))))
                del parts[0]  # first element is frame number, we don't need this
                for i, p in enumerate(parts): # normalizing step
                    parts[i] = p + 2000 / 4000
                sequence.append(parts)

            if len(sequence[0]) == 99:  # most frames have 99 points, we need a fixed input size for the network
                for i in range(0, len(sequence) - self.cut_length, self.step_size):
                    s = {"emotion": emotion, "motion": motion, "order": order,
                         "sequence": sequence[i:i + self.cut_length]}
                    self.samples["data"].append(s)

        # print(str(len(self.samples["data"])) + " samples loaded.")
        return self.samples

    def make_batch(self):
        """
        Creates a batch of certain length from the training data. Keeps remaining unused samples in case a file has
        more samples than fit in a single batch.
        :return: batch of length self.batch_size
        """
        batch = []
        if self.unused_samples:
            for s in self.unused_samples["data"]:
                if len(batch) >= self.batch_size:
                    return batch
                batch.append(s)
        if len(self.filenames) == 0:
            print("No more training samples... Stopping...")
            return None
        for f in range(len(self.filenames)):
            self.unused_samples = self.read_data(self.filenames[f])
            self.filenames.pop(f)
            for s in self.unused_samples["data"]:
                if len(batch) >= self.batch_size:
                    return batch
                batch.append(s)

    def split_batches(self, batch):
        """
        Splits a batch into two seperate batches for the motion and the label, so that it fits into our network input
        :param batch: the batch to be split
        :return: motion batch, label batch
        """
        motion_batch = []
        label_batch = []
        for s in batch:
            motion_batch.append(s["sequence"])
            label_batch.append(s["emotion"] + s["motion"])
        return motion_batch, label_batch

    def save_models(self, id):
        """
        Saves the current state of all models
        :param id: unique extension to the filename
        """
        with open("casae" + id + ".yaml", "w") as file:
            file.write(self.casae.to_yaml())
        with open("encoder" + id + ".yaml", "w") as file:
            file.write(self.encoder.to_yaml())
        with open("decoder" + id + ".yaml", "w") as file:
            file.write(self.decoder.to_yaml())
        with open("discmt" + id + ".yaml", "w") as file:
            file.write(self.discmt.to_yaml())

    def load_models(self, filepaths):
        """
        loads weights for the models
        :param filepaths: A list of four paths to saved weights, order: casae, encoder, decoder, discmt
        """
        self.casae.load_weights(filepaths[0])
        self.encoder.load_weights(filepaths[1])
        self.decoder.load_weights(filepaths[2])
        self.discmt.load_weights(filepaths[3])

    def write_data(self, data, path, file):

        filestring = \
            """$Comments \t File auto-generated with CASAE 
                        
$Filename %s
$Datetime %s

$FirstFrame 1
$LastFrame %d
$NumFrames %d
$NumMarkers 33
$CaptureRate 60
$Rate 60

$Order
LFHD RFHD LBHD RBHD C7 CLAV STRN LSHO LELB LWRA LWRB LFIN RSHO RELB RWRA RWRB RFIN T10 SACR LFWT RFWT LBWT RBWT LKNE RKNE LANK RANK LHEL RHEL LMT5 RMT5 LTOE RTOE 

$Points
""" % (file, datetime.now(), data.shape[0], data.shape[0])
        index = 1
        for frame in data:
            filestring = filestring + str(index) + "\t"
            for value in frame: # denormalizing
                value = 4000 * value - 2000
                filestring= filestring + str(value) + "\t"
            filestring = filestring + "\n\n"
            index += 1
        with open(os.path.join(path, file), "w") as file:
            file.write(filestring)
        return

    def train(self):
        """
        Trains the autoencoder
        """
        self.store_filenames()
        for i in range(self.epochs):
            print("epoch:", i + 1, "/", self.epochs)
            batch = self.make_batch()
            if batch == None:
                return -1
            motion_batch, label_batch = self.split_batches(batch)
            latent_codes = self.encoder.predict(x=[np.asarray(motion_batch), np.asarray(label_batch)],
                                                batch_size=self.batch_size)
            random_noise = np.random.multivariate_normal(np.zeros(self.latent_dim), np.eye(N=self.latent_dim) * 1.0,
                                                         size=self.batch_size)
            X = np.concatenate([latent_codes, random_noise], axis=0)
            Y = [1.0] * self.batch_size + [0.0] * self.batch_size
            self.discmt.trainable = True
            self.discmt.train_on_batch(x=np.asarray(X), y=np.asarray(Y))
            Y_hat = np.asarray([1.] * self.batch_size, dtype=np.float32)
            self.casae.train_on_batch(x=[np.asarray(motion_batch), np.asarray(label_batch), np.asarray(label_batch)],
                                      y={"Decoder": np.asarray(motion_batch), "Discmt": np.asarray(Y_hat)})


    def predict(self, target):
        noise = np.random.multivariate_normal(np.zeros(self.latent_dim), np.eye(N=self.latent_dim) * 1.0,
                                              size=self.batch_size)
        target = np.repeat(np.asarray(target), noise.shape[0]).reshape([10, 10])
        return self.decoder.predict(x=[noise, target])


generator = ConditionalGenerator()
generator.store_filenames()
generator.decoder.summary()
generator.train()
generator.save_models("2")
samples = generator.predict([0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
for i, s in enumerate(samples):
 generator.write_data(s,  ".", "sample_10k_angry_knock%d.csm"%i)
