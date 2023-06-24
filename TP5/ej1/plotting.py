import numpy as np
import matplotlib.pyplot as plt
import json
import utils_1
import sys
sys.path.append("..")
from multilayer.multilayer_perceptron import *
from multilayer.layer import Layer
from src.utils import *

# plotea para distinta cantidad de epocas el promedio del error en epocas (por ej para 10 epocas tengo un array de errores por epoca -> un promedio de esos valores)
def plot_variation_epochs():

    epochs_variation = [10,100,1000,10000]
    epochs_variation_labels = ["10","100","1000","10000"]

    with open('./config.json', 'r') as f:
        data = json.load(f)
        f.close()

    momentum, eta, epochs = utils_1.getDataFromFile(data)

    x = np.array(get_input(1))
    text_names = get_header(1)

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)

    x = transform(x)

    layer1 = Layer(20, 35, activation="tanh")
    #layer2 = Layer(20, activation="tanh")
    layer3 = Layer(2, activation="tanh")
    layer4 = Layer(20, activation="tanh")
    #layer5 = Layer(30, activation="tanh")
    layer6 = Layer(35, activation="tanh")

    layers = [layer1, layer3, layer4, layer6]

    error = {}

    for i in range(len(epochs_variation)):

        error[epochs_variation[i]] = []

        encoderDecoder = MultilayerPerceptron(layers, init_layers=True, momentum=momentum, eta=eta)

        min_error, errors, epochs, training_accuracies = encoderDecoder.train(x, x, iterations_qty=epochs_variation[i], adaptative_eta=False)

        encoder = MultilayerPerceptron(encoderDecoder.neuron_layers[0:int(len(layers)/2)], init_layers=False)

        decoder = MultilayerPerceptron(encoderDecoder.neuron_layers[int(len(layers)/2):], init_layers=False)

        # print(error)

        for j in range(len(x)):
            to_predict = x[j, :]
            # print("Predict: \n",to_predict)
            encoded = encoder.predict(to_predict)
            decoded = decoder.predict(encoded)
            # print("Decoded: \n",decoded)
            # print(i, calculate_error(to_predict, decoded))
            error[epochs_variation[i]].append(calculate_error(to_predict, decoded))
    print(error)
    x_plot = epochs_variation_labels
    y_plot = [np.mean(values) for values in error.values()]
    # for key, values in error.items():
    #     x_plot.extend([key] * len(values))
    #     y_plot.extend(np.mean(values))

    # Plot scatter
    plt.scatter(x_plot, y_plot)

    # Customize the plot
    plt.title("Mean error for different epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Mean error")

    # Show the plot
    plt.show()

#plote el error por epocas para 10000 epocas 
def plot_error_by_epochs():

    with open('./config.json', 'r') as f:
        data = json.load(f)
        f.close()

    momentum, eta, epochs = utils_1.getDataFromFile(data)

    x = np.array(get_input(1))
    text_names = get_header(1)

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)

    x = transform(x)

    layer1 = Layer(20, 35, activation="tanh")
    #layer2 = Layer(20, activation="tanh")
    layer3 = Layer(2, activation="tanh")
    layer4 = Layer(20, activation="tanh")
    #layer5 = Layer(30, activation="tanh")
    layer6 = Layer(35, activation="tanh")

    layers = [layer1, layer3, layer4, layer6]

    error = []

    encoderDecoder = MultilayerPerceptron(layers, init_layers=True, momentum=momentum, eta=eta)

    min_error, errors, epochs, training_accuracies = encoderDecoder.train(x, x, iterations_qty=10000, adaptative_eta=False)

    encoder = MultilayerPerceptron(encoderDecoder.neuron_layers[0:int(len(layers)/2)], init_layers=False)

    decoder = MultilayerPerceptron(encoderDecoder.neuron_layers[int(len(layers)/2):], init_layers=False)

    # print(error)

    for j in range(len(x)):
        to_predict = x[j, :]
        # print("Predict: \n",to_predict)
        encoded = encoder.predict(to_predict)
        decoded = decoder.predict(encoded)
        # print("Decoded: \n",decoded)
        # print(i, calculate_error(to_predict, decoded))
        error.append(calculate_error(to_predict, decoded))

    print(errors)
    x_plot = [i for i in range(10000)]
    # y_plot = errors #error
    y_plot = [np.mean(values) for values in error.values()] #mean error


    # Plot scatter
    plt.scatter(x_plot, y_plot)

    # Customize the plot
    plt.title("Error by epochs for 10000 epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Error")

    # Show the plot
    plt.show()

def calculate_error(to_predict, decoded):
    error = 0
    for i in range(len(to_predict)):
        error += (to_predict[i] - decoded[i])**2
    return error/len(to_predict)

def vary_hidden_layer():
    with open('./config.json', 'r') as f:
        data = json.load(f)
        f.close()

    momentum, eta, epochs = utils_1.getDataFromFile(data)

    x = np.array(get_input(1))
    text_names = get_header(1)

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)

    x = transform(x)

    layer1 = Layer(20, 35, activation="tanh")
    layer2 = Layer(10, activation="tanh")
    layer3 = Layer(2, activation="tanh")
    layer4 = Layer(10, activation="tanh")
    layer5 = Layer(20, activation="tanh")
    layer6 = Layer(35, activation="tanh")

    layers = [layer1,layer2, layer3, layer4, layer5,layer6]

    error = []

    encoderDecoder = MultilayerPerceptron(layers, init_layers=True, momentum=momentum, eta=eta)

    min_error, errors, epochs, training_accuracies = encoderDecoder.train(x, x, iterations_qty=10000, adaptative_eta=False)

    encoder = MultilayerPerceptron(encoderDecoder.neuron_layers[0:int(len(layers)/2)], init_layers=False)

    decoder = MultilayerPerceptron(encoderDecoder.neuron_layers[int(len(layers)/2):], init_layers=False)

    aux_1 = []
    aux_2 = []

    for j in range(len(x)):
        to_predict = x[j, :]
        # print("Predict: \n",to_predict)
        encoded = encoder.predict(to_predict)
        #print(encoded)
        decoded = decoder.predict(encoded)
        # print("Decoded: \n",decoded)
        # print(i, calculate_error(to_predict, decoded))
        graph_digits(to_predict, decoded)
        aux_1.append(encoded[0])
        aux_2.append(encoded[1])

    # plt.xlim([-1.1, 1.1])
    # plt.ylim([-1.1, 1.1])
    # for i, txt in enumerate(text_names):
    #     plt.annotate(txt, (aux_1[i], aux_2[i]))
    # plt.scatter(aux_1, aux_2)
    plt.show()

def plot_vary_eta():
    errors = {}
    with open('./config.json', 'r') as f:
        data = json.load(f)
        f.close()

    momentum, eta, epochs = utils_1.getDataFromFile(data)

    etas = [0.05, 0.005, 0.0005, 0.00005]
    etas_label = ["0.05", "0.005", "0.0005", "0.00005"]

    # for j in range(len(etas)):
        
    errors[etas_label[0]] = []
    for i in range(10):

        x = np.array(get_input(2))
        text_names = get_header(2)

        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)

        x = transform(x)

        # layer0 = Layer(2, 35, activation="tanh")
        layer1 = Layer(20, 35, activation="tanh")
        # layer2 = Layer(10, activation="tanh")
        layer3 = Layer(2, activation="tanh")
        # layer4 = Layer(10, activation="tanh")
        layer5 = Layer(20, activation="tanh")
        layer6 = Layer(35, activation="tanh")

        # layers = [layer0, layer6]
        # layers = [layer1, layer2, layer3, layer4, layer5, layer6]
        layers = [layer1, layer3, layer5, layer6]

        encoderDecoder = MultilayerPerceptron(layers, init_layers=True, momentum=momentum, eta=0.05)

        min_error, error, epochs, training_accuracies = encoderDecoder.train(x, x, iterations_qty=epochs, adaptative_eta=False)

        encoder = MultilayerPerceptron(encoderDecoder.neuron_layers[0:int(len(layers)/2)], init_layers=False)

        decoder = MultilayerPerceptron(encoderDecoder.neuron_layers[int(len(layers)/2):], init_layers=False)

        for i in range(len(x)):
            to_predict = x[i, :]
            encoded = encoder.predict(to_predict)
            decoded = decoder.predict(encoded)
            # aux_1.append(encoded[0])
            # aux_2.append(encoded[1])
            # graph_digits(to_predict, decoded)
            # error.append(calculate_error(to_predict, decoded))
            errors[etas_label[0]].append(calculate_error(to_predict, decoded))
    print(errors)

if __name__ == '__main__':
    vary_hidden_layer()
    # error = {'0.05': [0.4027798048112663, 0.533993995500369, 0.8192826341049827, 0.44731331514798484, 0.5623014419561927, 0.678368402960821, 0.0014143099799703467, 0.7562962959142747, 0.5680690885302151, 0.7256672608590911, 0.9205715474860201, 0.5208596277017233, 0.6849061143444675, 0.6258377835352901, 0.5425865576302099, 0.5364711010554799, 0.6190931631566314, 0.6449352803602254, 0.6956309239863755, 0.8188709982124034, 0.4971433947873373, 0.7079322469256043, 0.7419607593491967, 0.8489552009784853, 0.7145772891203312, 0.5280695598277761, 0.656558206291474, 0.45433999029403316, 0.5820328205912385, 0.4912888860472837, 0.8153501789397943, 0.5488292771433588, 0.39083981955831476, 0.6578849067526088, 0.7050047634448503, 0.7300385035599337, 0.8665867827825082, 0.6541565704599899, 0.48572531909812794, 0.5790785190437933, 0.6145947017591802, 0.49568295727884604, 0.4652549778454935, 0.6985932370284521, 0.6091786432843446, 0.5223132017084676, 0.6835904928453335, 0.33376058781319984, 0.5110961381639593, 0.5684224529883753, 0.5398422987608369, 0.7560917006856394, 0.41440682371286763, 0.49993262136911176, 0.5939901840506825, 0.7234673731916217, 0.7763439337671507, 0.5337092165788918, 0.777528795965094, 0.5530885522521534, 0.40805352343795853, 0.6927731423984255, 0.429355270989731, 0.6819033306687476, 0.7509171808078847, 0.5813202340798376, 0.5334136267887982, 0.6614962192289153, 0.6894035271596753, 0.67198508883806, 0.49129905569533594, 0.6225842741050273, 0.6306832378646572, 0.3806149985795381, 0.8600933404811976, 0.7360575375279543, 0.606050812033934, 0.5734245733475324, 0.6036555174433094, 0.6627457651927188, 0.5677162653583802, 0.8700355561631986, 0.561633214086374, 0.7833382601600141, 0.48497190524411593, 0.49454839283463364, 0.6574301300862095, 0.5236938046914654, 0.5333440420980645, 0.7604118084410963, 0.8826999611125945, 0.6362741685410599, 0.7153036776613335, 0.3563900534001549, 0.6617236777983411, 0.5913788932094928, 0.7220285671383012, 0.5913598841911246, 0.5357981694153757, 0.5439535690534311], '0.005': [0.4393343048210948, 0.39445999421161354, 0.3974063081505452, 0.5359095285496814, 0.13210218556309242, 0.6190694476940456, 0.4348646073587571, 0.3889455943686433, 0.5194348677493028, 0.3699830009064918, 0.41218301988053196, 0.582182926078902, 0.5247449246426795, 0.5187375433614365, 0.4499242207747742, 0.34737063088021386, 0.5316298278462638, 0.4063134110451187, 0.520233395046167, 0.3730323497136885, 0.5463980014711938, 0.5193505627311815, 0.3934993419866548, 0.3451791328418694, 0.3315606875607019, 0.510660526702396, 0.4429309416636089, 0.27095941163553233, 0.14776414406669505, 0.5299718050920494, 0.5375937137571926, 0.4217470747690439, 0.3369900909994034, 0.3859802151079281, 0.39907206812717233, 0.4879515605001102, 0.5304409163253366, 0.5953497291977107, 0.396395620234219, 0.4998900867268572, 0.03752956060523061, 0.49215555958895996, 0.36284619157795783, 0.3482852183980119, 0.39125813404971693, 0.5595474780874459, 0.42107352769604667, 0.4899470113882741, 0.32612321663000715, 0.3696577317292158, 0.4827377358677541, 0.40899137578913813, 0.38836630956619617, 0.5140275783192297, 0.398265511158938, 0.3476159449155931, 0.453755671731167, 0.40130805078281595, 0.3715544535462922, 0.678913891608139, 0.3646242727859932, 0.40004115625069375, 0.36199374430145387, 0.5513745771616019, 0.5054418601228112, 0.409308398945389, 0.46774538055359327, 0.5940324318271846, 0.4056597058913429, 0.05229923321824624, 0.5060302886319619, 0.4740394715048231, 0.5991140439708434, 0.475144640938946, 0.4897656779911768, 0.42804683047780473, 0.41943810445248025, 0.19400428967020017, 0.3664391979122713, 0.5258586704723862, 0.4802938714801934, 0.5134664769485464, 0.5678902638865303, 0.4571277860153927, 0.2575624477977234, 0.5763277870706465, 0.35868501666455677, 0.6069628906370979, 0.5530524106486527, 0.6444337454494296, 0.6359929048550672, 0.5014944787490412, 0.32271521200195785, 0.42577604935444474, 0.4108750949696278, 0.21197462903761247, 0.18766693189990963, 0.5738221235744672, 0.3999583340421117, 0.44495549112314775], '0.0005': [0.5371605367254252, 0.7174377851441275, 0.664016103047206, 0.4111038138456663, 0.4846521149652128, 0.48153640396714326, 0.7599601709710414, 0.49511282263857553, 1.0733408163345333, 0.40925264967609293, 0.4682237047965114, 0.638908715942607, 0.7154367203274912, 0.5731529468120854, 0.5019504208047526, 0.8164840339721081, 0.5456620288213981, 0.6501199385845078, 0.4990084637963615, 0.663463402715214, 0.2750478521551424, 0.9905867658562718, 0.5747307374470207, 0.32018672999705045, 0.4031193166769408, 0.504127265242968, 1.0606059672211448, 0.8144187751869749, 1.119485908273374, 0.6751858941755355, 0.8964388239103533, 0.641656803915198, 0.5211265968004021, 0.5901603667357961, 0.9340056412139077, 0.5103495010748247, 0.5573385312510774, 0.3917829042505012, 0.6219921614056525, 0.5375851005267399, 0.9375092156401273, 0.6528491337053576, 0.522516306303984, 0.5820912466186065, 0.8543823085967001, 0.39308001387188646, 1.103930633798457, 0.6675115138296991, 0.6410685327972333, 0.6054095620908173, 0.4837789263742573, 0.566481003485224, 0.46042136980353404, 0.6128533372769012, 0.5291964561485756, 0.6209239474581254, 0.35456505583071846, 0.4438529206302527, 0.55108202716444, 0.82717589683985, 0.7029994613461873, 0.26592635898952227, 0.8982649820949181, 1.0407963292468774, 0.4984033012452934, 0.7937760792205946, 0.5311779550739711, 0.8109960186830375, 0.8024428274806961, 0.4249682885168304, 1.0434615839140915, 0.5379965787326381, 0.6997671153827438, 0.6818268579931609, 0.9679439456707826, 0.6627713716444621, 0.49363609034219796, 1.1549565186806374, 0.572544548828119, 0.41446855223215656, 0.5196906450928993, 0.5765030098590963, 1.1368693416820854, 0.7958144514214066, 0.6162019863703225, 0.7271009151285937, 0.6422668697460775, 0.5438606925498354, 0.6272218145603462, 0.8053976102562957, 0.5265216323277463, 0.49308103576224543, 0.5494675036148653, 0.654320360880031, 0.8223081948036254, 0.5453601488242528, 0.6134263279008038, 0.8397083838927358, 1.245890185574293, 0.7731302901242865], '0.00005': [1.5432377163578037, 1.1457561511002576, 1.4090617737836262, 0.8482996297654911, 1.2836324915586454, 1.2819949335141838, 1.1449123513364998, 1.6366587850014696, 1.0727137215267226, 1.1895324991497749, 0.9727106237860397, 1.3690101647462134, 0.9866034406299256, 1.0743204978937504, 1.095700267143158, 0.9979080738236592, 1.0397420879929926, 1.0039400279210857, 1.0596807442013008, 1.9360112758756125, 1.1664493865985834, 1.440552668475297, 1.2961558907314956, 1.3248434902837012, 1.565343291776739, 1.5254727139713433, 1.619361764579427, 1.3868285742821975, 0.9812254960433575, 1.3017199954963719, 1.3625731951439048, 1.217267269429461, 1.582083579051845, 1.3592433036365537, 1.238770197782437, 0.9030277301716022, 1.3964446002059265, 1.5369123974081014, 0.8093507099953987, 0.9309784717355623, 1.085039343382779, 1.438349876424591, 1.2831186466728874, 1.4802950048957417, 1.070149952088584, 1.0724446872232807, 0.7438132429911476, 1.2492485053458444, 1.106314708111915, 1.6770784145161806, 1.2722667689791718, 0.9150962327959123, 1.566251325826936, 0.8520897955857418, 1.3664573224588203, 1.3001688378479728, 0.8279766926540771, 1.8589153256477131, 1.5892071481025276, 1.2088545991532533, 1.65369914415631, 0.9400612531515473, 1.51904401393054, 1.6130880396086684, 1.1433315225367555, 0.9753146196823679, 1.015001147789033, 1.1814122849895359, 1.0653229330511103, 1.4007963810200772, 1.3157218953007315, 0.975668009797493, 0.9393204077347244, 0.7933360495932911, 1.1192244049155444, 1.0605600448157266, 1.1819777392471753, 1.365901963328192, 1.7373749219526808, 1.677885763440646, 1.0476920959289155, 1.2372275776347164, 1.3417477976500556, 1.538617888995313, 1.7955444878543023, 1.6254934936217407, 0.7249473405605754, 1.3597156079547597, 1.3654310247540298, 1.7698629629771734, 1.4948773659266703, 0.8901873704867507, 1.2979261621674543, 1.619192763217591, 1.2119932984788926, 1.4720140661138292, 1.9572955931090352, 1.592908447379248, 1.1680432812222574, 1.3385793165378206]}
    # values = [0.6550986081116045,0.6117435579063374, 0.43492943796463396,1.2752250889722587]
    # # print(values)
    # std = [0.20743298441849578, 0.1407801641547856, 0.11919968722956936, 0.2794685809356365]
    # # print(std)
    # # plt.bar(error.keys(), values, width=0.4)
    # # plt.show()

    # # Build the plot
    # fig, ax = plt.subplots()
    # ax.bar(error.keys(), values, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
    # ax.set_ylabel('Mean error')
    # ax.set_xlabel('ETA')
    # ax.set_xticklabels(error.keys())
    # ax.set_title('Mean error for diffent ETAs')
    # # ax.yaxis.grid(True)

    # # Save the figure and show
    # plt.tight_layout()
    # # plt.savefig('bar_plot_with_error_bars.png')
    # plt.show()