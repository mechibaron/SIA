from matplotlib import pyplot as plt
from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect 
import numpy as np 
import pandas as pd

if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")
    range_num=100
    # snorlax = factory.create("snorlax", 100, StatusEffect.NONE, 1)
    # print("No noise: ", attempt_catch(snorlax, "heavyball"))
    pokeballs = ['heavyball','fastball','pokeball','ultraball']
    pokemons = ['jolteon', 'caterpie', 'snorlax', 'onix', 'mewtwo']     
    pokemonandprob = pd.DataFrame({'heavyball' : [0, 0, 0, 0, 0, 0],
                     'fastball': [0, 0, 0, 0, 0, 0],
                     'pokeball': [0, 0, 0, 0, 0, 0],
                     'ultraball': [0, 0, 0, 0, 0, 0]},
                    index=('POISON', 'BURN', 'PARALYSIS', 'SLEEP', 'FREEZE','NONE'))   
    # print(pokemonandprob)
    status = [StatusEffect.POISON, StatusEffect.BURN,StatusEffect.PARALYSIS,StatusEffect.SLEEP,StatusEffect.FREEZE,StatusEffect.NONE]

    i = 0
    for pokemon in range(len(status)):
        myPokemon = factory.create('jolteon', 100, status[pokemon], 0)
        successPokemon = 0
        print()
        print("status effect: ", pokemon)
        for pokeball in pokeballs:
            print("\tFor Pokeball:", pokeball)
            successPokeball = 0
            for _ in range(range_num):
                attempt = attempt_catch(myPokemon, pokeball, 0.15)
                if (attempt[0]) : 
                    successPokemon+=1
                    successPokeball+=1
                # print("\t\tNoisy: ", attempt)
            print("\tSuccess Pokeball: ", successPokeball)
            successprob = successPokeball/range_num
            print("\tPokeball efectivity: " , successprob)
            pokemonandprob[pokeball][i] = successPokeball
        print("Success Pokemon: ", successPokemon)
        i+=1

    print(pokemonandprob)
    x = np.arange(len(pokemonandprob.index))
    width =  0.2
    plt.bar(x - width, pokemonandprob.heavyball, width=width, label="Heavyball")
    plt.bar(x, pokemonandprob.fastball, width=width, label="Fastball")
    plt.bar(x + width, pokemonandprob.pokeball, width=width, label="Pokeball")
    plt.bar(x + (2*width), pokemonandprob.ultraball, width=width, label="Ultraball")
    plt.xticks(x, pokemonandprob.index)
    plt.legend(loc='best')
    plt.show()
    # for i in range(len(pokeballs)) : 
    #     print ("Promedy Pokeball success for : ", pokeballs[i], ";", prom_pokebolls[i]/100)    