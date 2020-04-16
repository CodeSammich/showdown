import asyncio
import numpy as np
import json
from copy import deepcopy
from  async_timeout import timeout
import matplotlib.pyplot as plt

from environs import Env

import constants
import config
from config import init_logging, Config
import logging

from teams import load_team
#from showdown.run_battle import pokemon_battle
from showdown.run_train import pokemon_battle
from showdown.websocket_client import PSWebsocketClient


from data import all_move_json
from data import pokedex
from data.mods.apply_mods import apply_mods
from showdown.battle_bots.nn_bot.DQNAgent import Agent
from showdown.battle_bots.nn_bot.deep_q_network import DeepQNetwork
import time
import torch
from showdown.battle_bots.nn_bot.deep_q_network import DeepQNetwork
from showdown.engine.evaluate import evaluate

logger = logging.getLogger(__name__)

def create_challenge_bot(agent):
    """
    agent: specify str agent name (i.e rand_bot) or a DQNAgent. If DQNAgent will choose nn_bot
    """
    conf = Config()
    if type(agent) == type(""):
        conf.battle_bot_module = "rand_max"
    else:
        conf.battle_bot_module = "nn_bot"
    conf.save_replay = config.save_replay
    conf.use_relative_weights = config.use_relative_weights
    conf.gambit_exe_path = config.gambit_exe_path
    conf.search_depth = config.search_depth
    conf.websocket_uri = "localhost:8000"
    conf.username = "cbninjask5uber"
    conf.password = "aiaccount1"
    conf.bot_mode = "CHALLENGE_USER"
    conf.team_name = "gen8/ou/band_toad"
    conf.pokemon_mode = "gen8ou"
    conf.run_count = 5
    conf.user_to_challenge = "AcceptGary"
    conf.LOG_LEVEL = 'DEBUG'
    return conf

def create_accept_bot(agent):
    """
    agent: specify str agent name (i.e rand_bot) or a DQNAgent. If DQNAgent will choose nn_bot
    """
    conf = Config()
    if type(agent) == type(""):
        conf.battle_bot_module = "rand_bot"
    else:
        conf.battle_bot_module = "nn_bot"
    conf.save_replay = config.save_replay
    conf.use_relative_weights = config.use_relative_weights
    conf.gambit_exe_path = config.gambit_exe_path
    conf.search_depth = config.search_depth
    conf.websocket_uri = "localhost:8000"
    conf.username = "AcceptGary"
    conf.password = "password"
    conf.bot_mode = "ACCEPT_CHALLENGE"
    conf.team_name = "gen8/ou/band_toad"
    # conf.team_name = "gen8/ou/clef_sand"
    conf.pokemon_mode = "gen8ou"
    conf.run_count = 5
    conf.LOG_LEVEL = 'DEBUG'
    return conf

def check_dictionaries_are_unmodified(original_pokedex, original_move_json):
    # The bot should not modify the data dictionaries
    # This is a "just-in-case" check to make sure and will stop the bot if it mutates either of them
    if original_move_json != all_move_json:
        logger.critical("Move JSON changed!\nDumping modified version to `modified_moves.json`")
        with open("modified_moves.json", 'w') as f:
            json.dump(all_move_json, f, indent=4)
        exit(1)
    else:
        logger.debug("Move JSON unmodified!")

    if original_pokedex != pokedex:
        logger.critical("Pokedex JSON changed!\nDumping modified version to `modified_pokedex.json`")
        with open("modified_pokedex.json", 'w') as f:
            json.dump(pokedex, f, indent=4)
        exit(1)
    else:
        logger.debug("Pokedex JSON unmodified!")


async def showdown(accept, agent=None):
    """
    Will run through number of battles specified in config

    accept: boolean. If accept is true will create an accept_bot
    agent: specify str agent name (i.e rand_bot) or a DQNAgent. If DQNAgent will choose nn_bot
    """
    if accept:
        conf = create_accept_bot(agent)  # accept gary
        #await asyncio.sleep(10)  # ensure that challenge bot has time to be created first
    else:
        await asyncio.sleep(4)  # ensure that challenge bot has time to be created first
        conf = create_challenge_bot(agent)  # cbninjask5uber

    config = conf
    apply_mods(config.pokemon_mode)

    original_pokedex = deepcopy(pokedex)
    original_move_json = deepcopy(all_move_json)

    ps_websocket_client = await PSWebsocketClient.create(config.username, config.password, config.websocket_uri)
    await ps_websocket_client.login()

    wins = 0
    losses = 0

    team = load_team(config.team_name)
    if config.bot_mode == constants.CHALLENGE_USER:
        await ps_websocket_client.challenge_user(config.user_to_challenge, config.pokemon_mode, team)
    elif config.bot_mode == constants.ACCEPT_CHALLENGE:
        await ps_websocket_client.accept_challenge(config.pokemon_mode, team)
    elif config.bot_mode == constants.SEARCH_LADDER:
        await ps_websocket_client.search_for_match(config.pokemon_mode, team)
    else:
        raise ValueError("Invalid Bot Mode")

    if type(agent) == type(""):
        winner = await pokemon_battle(ps_websocket_client, config.pokemon_mode, config, agent = None)
    else:
        winner = await pokemon_battle(ps_websocket_client, config.pokemon_mode, config, agent)

    if winner == config.username:
        finalReward = 1000
        wins += 1
    else:
        finalReward = -1000
        losses += 1

    if type(agent) != type("str"):
        agent.step(agent.previous_state, agent.previous_action, finalReward, torch.zeros(8175), True)

    # if accept:
    #     logger.debug("W: {}\tL: {}".format(wins, losses))
    #     score = agent.memory.memory.copy().pop()[2]
    #     total_score += score + finalReward
    #     logger.debug("End Score: {}".format(score)) #
    #     agent.step(agent.previous_state, agent.previous_action, finalReward, torch.zeros(8175), True)
    # else:
    #     logger.debug("W: {}\tL: {}".format(wins, losses))


    check_dictionaries_are_unmodified(original_pokedex, original_move_json)
    return winner == config.username

    # battles_run += 1
    # if battles_run >= config.run_count:
    #     break

    # if accept:
    #     logger.critical("W: {}\tL: {}".format(wins, losses))
    #     logger.critical("End Score: {}".format(total_score/(wins + losses)))
    #
    # return
async def train_episode(agent1, agent2):
    """
    Goal of function is to put a neural network bot versus a hard coded bot
    """
    return await asyncio.gather(showdown(accept=True, agent=agent1), showdown(accept=False, agent=agent2))

async def main():
    """Call this code only once"""
    init_logging("DEBUG")

    """Training params"""
    episodes = 100
    merge_networks_time = 20  # run this many times and then merge multiple agents TODO

    """Performance Params"""
    eval_time = 30 # evals the network every eval_time steps
    eval_run_battles = 3  # runs this many battles to determine performance against
    eval_opponent = "rand_bot"  # what is the neural network evaluating against
    winPercList = []
    episodeList = []

    seed = np.random.randint(0, 50)

    """Network Params"""
    state_size = 8175
    actions = 9
    agent1 = Agent(state_size, actions, seed)
    agent1.train()  # agent is in training mode
    agent2 = "rand_bot" # two agents should actually be playing against eachother

    # reinitialize and load weights
    checkpoint = torch.load('nn_bot_trained')
    model = DeepQNetwork(state_size, actions) 
    model.load_state_dict(checkpoint['local'])
    agent1.qnetwork_local = model
    model.load_state_dict(checkpoint['target'])
    agent1.qnetwork_target = model



    """main training loop"""
    for episode in range(episodes):
        print("episode", episode)

        start = time.time()
        await train_episode(agent1, agent2=agent2)  # can probably run some of these in parallel using gather
        print("Elapsed", time.time() - start)

        if (episode+1) % merge_networks_time == 0:
            pass  # TODO merge multiple agents

        if (episode + 1) % eval_time == 0:
            agent1.eval()
            wins = 0
            for i in range(eval_run_battles):
                print("Eval Number", i)
                agentWin, _ = await train_episode(agent1, agent2=eval_opponent)  # can probably run some of these in parallel using gather
                wins += agentWin
            episodeList.append(episode)
            winPercList.append(wins/eval_run_battles)
            agent1.train()  # allows the agent to train again

            torch.save({
                'local': agent1.qnetwork_local.state_dict(),
                'target': agent1.qnetwork_target.state_dict()
            }, "nn_bot_trained")

        plt.clf()
        plt.plot(episodeList,winPercList)
        plt.draw()
    print("done training")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
    print("done")