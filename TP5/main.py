import json
import utils
import font
import autoencoder

if __name__ == '__main__':
    with open('./config.json', 'r') as f:
        data = json.load(f)
        f.close()

    epochs, learning_rate, latent_size, batch_size = utils.get_data(data)
    input = font.Font3
    expected_output = font.expected_output

    encoder = autoencoder.Autoencoder(len(input), latent_size, input, expected_output, learning_rate, epochs, batch_size)
    encoder.build()
