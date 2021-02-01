
import tensorflow as tf

class WindowDataset():


    def __init__(self,  X_df = None, y_df = None,
                 window_size = 3000,
                 shift = 1000,
                 batch_size = 100,
                 shuffle_buffer = 10
                 ):
        self.X_df = X_df
        self.y_df = y_df
        self.window_size = window_size
        self.batch_size = batch_size
        self.shift = shift
        self.shuffle_buffer = shuffle_buffer
        self.windowed_dataset = None



    def window_dataset(self):
        if self.X_df is None or self.y_df is None:
            print('Please set x_df and y_df before running window_dataset()... aborting')
            return

        dataset = tf.data.Dataset.from_tensor_slices(self.X_df)
        dataset = dataset.window(self.window_size, shift=self.shift, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(self.window_size))

        dataset_y = tf.data.Dataset.from_tensor_slices(self.y_df)
        dataset_y = dataset_y.window(1, shift=1, drop_remainder=True)
        dataset_y = dataset_y.flat_map(lambda window: window.batch(1))

        # dataset = dataset.map(lambda window: (window, y_df.values.tolist()))
        # dataset = dataset.map(lambda x, y: (x , y),  dataset_y)

        dataset = tf.data.Dataset.zip((dataset, dataset_y))
        dataset = dataset.shuffle(self.shuffle_buffer)
        dataset = dataset.batch(self.batch_size).prefetch(1)
        self.windowed_dataset = dataset
        return dataset

    def get_window_data_batch_shape(self):
        if self.windowed_dataset is None:
            print('self.windowed_dataset is not created, run window_dataset to create.')
            return

        for x, y in self.windowed_dataset:
            print(len(x), '  per batch (', self.batch_size, ')')
            print(len(y), '  per batch (', self.batch_size, ')')

            print(len(x[0]), ' x length of 1 array in batch (', self.window_size, ')')  #
            print(len(y[0]), ' y length of 1 array in batch (1)')  #

            print(len(x[0][0]), ' x values per instance  (should be equal to the # of x columns)')

            print(len(y[0][0]), ' y values per instance  (should be equal to the # of y columns)')
            break